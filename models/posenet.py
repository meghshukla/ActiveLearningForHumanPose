'''
Baseline Architecture: Stacked Hourglass
https://github.com/princeton-vl/pytorch_stacked_hourglass
'''
import torch
from torch import nn
from .layers import Conv, Hourglass, Pool, Residual


class HeatmapLoss(torch.nn.Module):
    '''
    Compute the MSE loss for each set of heatmaps
    Loss reduced over all dimensions
    '''
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2)
        loss = loss.mean(dim=[1, 2, 3])
        return loss


class Merge(nn.Module):
    '''

    '''
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        '''

        :param nstack: (int) Number of stacks
        :param inp_dim: (int) Number of input channels for the Stacked Hourglass
        :param oup_dim: (int) Number of output channels for the Stacked Hourglass
        :param bn: (bool) Whether to perform Batch Normalization
        :param increase:
        :param kwargs:
        '''
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(inp_dim=3, out_dim=64, kernel_size=7, stride=2, bn=True, relu=True),
            Residual(inp_dim=64, out_dim=128),
            Pool(2, 2),
            Residual(inp_dim=128, out_dim=128),
            Residual(inp_dim=128, out_dim=inp_dim))
        
        self.hgs = nn.ModuleList(
            [nn.Sequential(Hourglass(n=4, f=inp_dim, bn=bn, increase=increase),
                           ) for i in range(nstack)])
        
        self.features = nn.ModuleList([nn.Sequential(Residual(inp_dim, inp_dim),
                                                     Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
                                                     ) for i in range(nstack)])
        
        self.outs = nn.ModuleList(
            [Conv(inp_dim=inp_dim, out_dim=oup_dim, kernel_size=1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack-1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        '''
        Constructing the Stacked Hourglass Posenet Model
        :param imgs:
        :return:
        '''
        x = imgs.permute(0, 3, 1, 2)   # x (input images) of size 1, 3, inpdim, inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
