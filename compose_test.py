import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
import cv2
from argparse import ArgumentParser

from models.rendering import *
from models.Block_NeRF import *

import metrics

import datasets.WaymoDataset
from train import *

torch.backends.cudnn.benchmark = True


def get_hparams():
    parser = ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default='result',
                        help='root directory of dataset')

    parser.add_argument('--root_dir', type=str,
                        default='data/WaymoDataset',
                        help='root directory of dataset')

    parser.add_argument('--N_samples', type=int, default=128,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', type=bool, default=False,  # 视差深度图
                        help='use disparity depth sampling')
    # NeRF-W
    parser.add_argument('--save_depth', type=bool, default=True,  # 视差深度图
                        help='Whether save the depth of the output')

    parser.add_argument('--cam_idx', type=int, default=10,
                        help='the index of the camera you want to inference,0~11, total 12 cameras')
    parser.add_argument('--ckpt_path', type=str, default="block_0.ckpt",
                        help='pretrained checkpoint path to load')
    '''
    parser.add_argument('--test_img_name', type=str, default="2082181371",  # 729712596 414587110 1202087606 116169437
                        help='the range to sample along the ray')
    '''

    return vars(parser.parse_args())


@torch.no_grad()
def batched_inference(model, embeddings,
                      rays, ts,
                      N_samples=128,
                      N_importance=128,
                      chunk=1024,
                      use_disp=False):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        result_chunk = render_rays(model, embeddings,
                                   rays[i:i + chunk], ts[i:i + chunk],
                                   N_samples=N_samples,
                                   N_importance=N_importance,
                                   chunk=chunk,
                                   type="test",
                                   use_disp=hparams['use_disp'])
        for k, v in result_chunk.items():
            results[k] += [v.cpu()]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results


def extract_cam_info(index, cam_infos):
    for i, cam_info in enumerate(cam_infos):
        if i == index:
            print(f"Now is inferencing the {cam_info} camera..")
            return cam_infos[cam_info]
    return None


def filter_Block(begin, blocks):
    block_filter = []
    for block in blocks:
        for element in blocks[block]['elements']:
            if element[0]==begin:
                block_filter.append(block)
    '''
    for block in blocks:
        centroid = blocks[block]['centroid'][1]
        if np.linalg.norm(np.array(centroid) - np.array(origin)) < 0.3:
            block_filter.append(block)
    '''
    return block_filter


def DistanceWeight(point, centroid, p=4):
    point = point.numpy()
    centroid = np.array(centroid)
    distance = np.linalg.norm(point - centroid)
    return (np.linalg.norm(point - centroid)) ** (-p)


def Inverse_Interpolation(model_result, W_H):  # 根据每帧的结果进行插值
    weights = []

    img_RGB={}
    img_DEPTH={}
    #   首先提取每一个模型的推理结果
    for block in model_result:
        block_RGB = np.clip(model_result[block]['rgb_fine'].view(H, W, 3).detach().numpy(), 0, 1)
        block_RGB = (block_RGB * 255).astype(np.uint8)
        img_RGB[block]=block_RGB

        block_depth =model_result[block]['depth_fine'].view(H, W).numpy()
        block_depth = np.nan_to_num(block_depth)  # change nan to 0
        mi = np.min(block_depth)  # get minimum depth
        ma = np.max(block_depth)
        block_depth = (block_depth - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        block_depth = (255 * block_depth).astype(np.uint8)
        img_DEPTH[block]=block_depth

        weights.append(model_result[block]['distance_weight'])

    weights = [weight / sum(weights) for weight in weights]
    img_pred = sum(weight * rgb for weight, rgb in zip(weights, img_RGB.values())).astype(np.uint8)
    img_depth = sum(weight * depth for weight, depth in zip(weights, img_DEPTH.values())).astype(np.uint8)

    img_RGB['compose']=img_pred
    img_DEPTH['compose']=img_depth

    return img_RGB,img_DEPTH

if __name__ == "__main__":
    torch.cuda.empty_cache()
    hparams = get_hparams()

    block_split_info = None
    # 1. extract each block's centroids
    with open(os.path.join(hparams['root_dir'], f'json/split_block_train.json'), 'r') as fp:  # 每一块的情况
        block_split_info = json.load(fp)
    centroids = []
    for block in block_split_info:
        centroids.append(block_split_info[block]['centroid'])

    # 2. 选取某个相机index
    with open(os.path.join(hparams['root_dir'], f'json/cam_info.json'), 'r') as fp:  # 每一块的情况
        cam_infos = json.load(fp)

    cam_infos = extract_cam_info(hparams['cam_idx'], cam_infos)  # 某个相机的行动轨迹
    cam_info_begin = cam_infos[:-1]  # 某条路径的起点
    cam_info_end = cam_info_begin[1:]  # 某条路径的终点
    print()

    block_model = ["block_0", "block_1"]#只渲染这两个模型

    os.makedirs(os.path.join('result', str(hparams['cam_idx'])), exist_ok=True)

    imgs = []
    imgs_depth = []
    for i in tqdm(range(len(cam_info_begin))):
        begin,end=cam_info_begin[i],cam_info_end[i]
        print(f"The {i}/{len(cam_info_begin)} stage of inferencing...")
        dataset = WaymoDataset(root_dir=hparams['root_dir'],
                               split='compose',
                               cam_begin=begin,
                               cam_end=end)
        # 每两个镜头之间有1个渲染
        for j in tqdm(range(len(dataset))):
            # 每一帧的结果由blocks共同决定
            batch = dataset[j]
            rays, ts = batch['rays'], batch['ts']

            W, H = batch['w_h']

            origin = rays[0, 0:3]
            # 1. 筛除不在覆盖范围内的block
            blocks = filter_Block(begin, block_split_info)
            print(f"The current view belongs to the block of {blocks}")
            # blocks为当前视野所位于的block区间
            model_result = {}
            # 判断每一个block的visibility
            for block in blocks:
                if block in block_model:
                    model = Block_NeRF_System.load_from_checkpoint(f"{block}.ckpt").cuda().eval()
                    models = {
                        "block_model": model.Block_NeRF,
                        "visibility_model": model.Visibility
                    }
                    print("Now is inferencing the {0}'s model".format(block))

                    ts[:]=find_idx_name(block_split_info[block]['elements'],begin)
                    # 将ts修改为当前block对应的ts
                    results = batched_inference(models, model.Embedding,
                                                rays.cuda(), ts.cuda(),
                                                use_disp=model.hparams['use_disp'],
                                                N_samples=hparams['N_samples'],
                                                N_importance=hparams['N_importance'])
                    print()
                    if results['transmittance_fine_vis'].mean() > 0.05:  # 当前block可见
                        # 计算权重 wi=(c,xi)^(-p)
                        results['distance_weight'] = DistanceWeight(point=origin,
                                                                    centroid=block_split_info[block]["centroid"][1],
                                                                    p=4)
                        model_result[block] = results
                    else:
                        print(f"As the {begin} can't be seen to the {block}, the mean of visibility is {results['transmittance_fine_vis'].mean()}")

                # 进行插值
            if len(model_result):
                RGB_compose, Depth_compose = Inverse_Interpolation(model_result, [W, H])
                imgs.append(RGB_compose['compose'])
                imgs_depth.append(Depth_compose['compose'])

                imageio.mimsave(os.path.join('result', str(hparams['cam_idx']), 'test.gif'),
                                imgs, fps=30)
                imageio.mimsave(os.path.join('result', str(hparams['cam_idx']), 'test_depth.gif'),
                                imgs_depth, fps=30)

                for RGB, Depth in zip(RGB_compose, Depth_compose):
                    imageio.imwrite(os.path.join('result', str(hparams['cam_idx']),
                                                 '{0}_{1}_{2}_{3}.png'.format(i, begin, end, RGB)), RGB_compose[RGB])
                    imageio.imwrite(os.path.join('result', str(hparams['cam_idx']),
                                                 '{0}_{1}_{2}_{3}_depth.png'.format(i, begin, end, Depth)),
                                    Depth_compose[Depth])
                print()

    imageio.mimsave(os.path.join('result', str(hparams['cam_idx']), 'test.gif'),
                    imgs, fps=30)
    imageio.mimsave(os.path.join('result', str(hparams['cam_idx']), 'test_depth.gif'),
                    imgs_depth, fps=30)
