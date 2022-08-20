'''
        loss += lambda(0.1) * (ret['rgb_coarse'] - rgbs[..., :3]) ** 2).mean()
        loss += (ret['rgb_fine'] - rgbs[..., :3]) ** 2).mean()

        loss += 1e-6 * (ret[f'visib_trans_fine'].squeeze(-1)-ret[f'trans_fine'].detach())**2).mean()
        loss += 1e-6 * (ret[f'visib_trans_coarse'].squeeze(-1)-ret[f'trans_coarse'].detach())**2).mean()
'''
import torch
from torch import nn


class BlockNeRFLoss(nn.Module):
    def __init__(self, λ_u=0.01,Visi_loss=1e-2):
        super(BlockNeRFLoss, self).__init__()
        self.λ_u = λ_u
        self.Visi_loss=Visi_loss

    def forward(self, inputs, targets):
        loss = {}
        # RGB
        loss['rgb_coarse'] = self.λ_u * ((inputs['rgb_coarse'] - targets[..., :3]) ** 2).mean()
        loss['rgb_fine'] = ((inputs['rgb_fine'] - targets[..., :3]) ** 2).mean()
        # visibility
        loss["transmittance_coarse"] = self.Visi_loss * ((inputs['transmittance_coarse_real'].detach() - inputs[
            'transmittance_coarse_vis'].squeeze()) ** 2).mean()
        loss["transmittance_fine"] = self.Visi_loss * ((inputs['transmittance_fine_real'].detach() - inputs[
            'transmittance_fine_vis'].squeeze()) ** 2).mean()

        return loss
