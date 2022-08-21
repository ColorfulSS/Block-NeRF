import os
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets.WaymoDataSet import *
# from datasets.WaymoDataset_test import *

from models.Block_NeRF import *
from models.loss import *
from models.rendering import *
from utils import *
from metrics import *

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger  # 保存在TensorBoard
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
                                #ModelCheckpoint：通过检测质量定期保存模型
                                #TQDMProgressBar：the default progress bar
from pytorch_lightning.loggers import TensorBoardLogger

import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='data/WaymoDataset',
                        help='root directory of dataset')
    parser.add_argument('--block_index', type=str,
                        default='block_0',  # 0.3,0.5 643张
                        help='root directory of dataset')

    parser.add_argument('--img_downscale', type=int, default=4,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--near', type=float, default=0.01,
                        help='the range to sample along the ray')
    parser.add_argument('--far', type=float, default=15,
                        help='the range to sample along the ray')

    parser.add_argument('--N_IPE_xyz', type=int, default=16,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_PE_dir_exposure', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    # NeRF-W
    parser.add_argument('--N_vocab', type=int, default=1500,
                        help='''number of vocabulary (number of images) 
                                        in the dataset for nn.Embedding''')
    parser.add_argument('--N_appearance', type=int, default=32,
                        help='number of embeddings for appearance')

    parser.add_argument('--Visi_loss', type=float, default=1e-2,
                        help='number of embeddings for appearance')

    parser.add_argument('--use_disp', type=bool, default=True,  # 视差深度图
                        help='use disparity depth sampling')

    parser.add_argument('--chunk', type=int, default=1024 * 16,
                        help='chunk to avoid OOM')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    # params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='print the progress bar every X steps')

    return vars(parser.parse_args())


class Block_NeRF_System(LightningModule):
    def __init__(self, hparams):
        super(Block_NeRF_System, self).__init__()
        self.save_hyperparameters(hparams)
        self.loss = BlockNeRFLoss(1e-2)#hparams['Visi_loss']

        self.xyz_IPE = InterPosEmbedding(hparams['N_IPE_xyz'])  # xyz的L=10
        self.dir_exposure_PE = PosEmbedding(
            hparams['N_PE_dir_exposure'])  # dir的L=4
        self.embedding_appearance = torch.nn.Embedding(
            hparams['N_vocab'], hparams['N_appearance'])

        self.Embedding = {'IPE': self.xyz_IPE,
                          'PE': self.dir_exposure_PE,
                          'appearance': self.embedding_appearance}

        self.Block_NeRF = Block_NeRF(in_channel_xyz=6 * hparams['N_IPE_xyz'],
                                     in_channel_dir=6 *
                                                    hparams['N_PE_dir_exposure'],
                                     in_channel_exposure=2 *
                                                         hparams['N_PE_dir_exposure'],
                                     in_channel_appearance=hparams['N_appearance'])

        self.Visibility = Visibility(in_channel_xyz=6 * hparams['N_IPE_xyz'],
                                     in_channel_dir=6 * hparams['N_PE_dir_exposure'])

        self.models_to_train = []
        self.models_to_train += [self.embedding_appearance]
        self.models_to_train += [self.Block_NeRF]
        self.models_to_train += [self.Visibility]

    def forward(self, rays, ts):
        B = rays.shape[0]
        model = {
            "block_model": self.Block_NeRF,
            "visibility_model": self.Visibility
        }

        results = defaultdict(list)
        for i in range(0, B, self.hparams['chunk']):
            rendered_ray_chunks = render_rays(model, self.Embedding,
                                              rays[i:i + self.hparams['chunk']],
                                              ts[i:i + self.hparams['chunk']],
                                              N_samples=self.hparams['N_samples'],
                                              N_importance=self.hparams['N_importance'],
                                              chunk=self.hparams['chunk'],
                                              type="train",
                                              use_disp=self.hparams['use_disp']
                                              )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def setup(self, stage):
        self.train_dataset = WaymoDataset(root_dir=hparams['root_dir'],
                                          split='train',
                                          block=hparams['block_index'],
                                          img_downscale=hparams['img_downscale'],
                                          near=hparams['near'],
                                          far=hparams['far'])
        self.val_dataset = WaymoDataset(root_dir=hparams['root_dir'],
                                        split='val',
                                        block=hparams['block_index'],
                                        img_downscale=hparams['img_downscale'],
                                        near=hparams['near'],
                                        far=hparams['far'])

        # dataset = WaymoDataset(root_dir="../data/WaymoDataset", split='train',block='block_0')

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams['batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results['rgb_fine'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):#每一个epoch校验一次
        rays, rgbs, ts = batch['rays'].squeeze(), batch['rgbs'].squeeze(), batch['ts'].squeeze()
        W,H=batch['w_h']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        if batch_nb == 0:
            img = results[f'rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_fine'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            #stack = torch.stack([img_gt, img])  # (3, 3, H, W)
            #不用*255吗
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results['rgb_fine'], rgbs)

        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[f'val_{k}']= v
        log['val_psnr']= psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = Block_NeRF_System(hparams)
    print(system.hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            'ckpts/{0}'.format(hparams['exp_name']), '{epoch:d}'),
        monitor='val/loss', mode='min',
        save_top_k=5)

    #pbar = TQDMProgressBar(refresh_rate=1)
    #callbacks = [checkpoint_callback, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams['block_index'],
                               default_hp_metric=False)

    '''
    logger = TestTubeLogger(save_dir="logs",
                            name=hparams['block_index'],
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)
    '''

    trainer = Trainer(max_epochs=hparams['num_epochs'],
                      precision=16,  # mix precision 半精度训练
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams['ckpt_path'],
                      logger=logger,
                      weights_summary='full',
                      enable_model_summary=True,  # 是否打印模型摘要
                      progress_bar_refresh_rate=hparams['refresh_every'],
                      gpus=hparams['num_gpus'],  # torch.cuda.device_count()
                      # accelerator='ddp' if hparams['num_gpus'] > 1 else 'auto',
                      accelerator='auto',
                      num_sanity_val_steps=1,#用于设置在开始训练前先进行num_sanity_val_steps个 batch的validation，以免你训练了一段时间，在校验的时候程序报错，导致浪费时间
                      # Sanity check runs n validation batches before starting the training routine
                      benchmark=True,## torch.backends.cudnn.benchmark，可以提升神经网络的运行速度
                      profiler="simple" if hparams['num_gpus'] == 1 else None,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams['num_gpus'] > 1 else None,

                      )

    trainer.fit(system)
    print("The best model is saved in the path: ",checkpoint_callback.best_model_path)


if __name__ == '__main__':
    hparams = get_opts()
    torch.cuda.empty_cache()
    main(hparams)
