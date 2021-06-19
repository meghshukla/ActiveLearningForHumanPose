'''
Baseline Architecture: Stacked Hourglass
https://github.com/princeton-vl/pytorch_stacked_hourglass
'''
import torch
from torch import nn
from .layers import Conv, Hourglass, Pool, Residual


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

        if torch.cuda.device_count() > 1:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        else:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:0')]

        self.cuda_devices = cuda_devices
        self.nstack = nstack

        self.pre = nn.Sequential(
            Conv(inp_dim=3, out_dim=64, kernel_size=7, stride=2, bn=True, relu=True),
            Residual(inp_dim=64, out_dim=128),
            Pool(2, 2),
            Residual(inp_dim=128, out_dim=128),
            Residual(inp_dim=128, out_dim=inp_dim)).cuda(cuda_devices[0])
        
        self.hgs = nn.ModuleList(
            [nn.Sequential(Hourglass(n=4, f=inp_dim, bn=bn, increase=increase),
                           ).cuda(cuda_devices[i]) for i in range(nstack)])
        
        self.features = nn.ModuleList([nn.Sequential(Residual(inp_dim, inp_dim),
                                                     Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
                                                     ).cuda(cuda_devices[i]) for i in range(nstack)])
        
        self.outs = nn.ModuleList(
            [Conv(inp_dim=inp_dim, out_dim=oup_dim, kernel_size=1, relu=False, bn=False).cuda(cuda_devices[i])
             for i in range(nstack)])

        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim).cuda(cuda_devices[i]) for i in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim).cuda(cuda_devices[i]) for i in range(nstack-1)])

        self.hg_avg_pool = nn.ModuleList([nn.AvgPool2d(kernel_size=(64, 64), stride=1).cuda(cuda_devices[i])
                                              for i in range(nstack)])
        self.global_avg_pool = nn.ModuleList([nn.AvgPool2d(kernel_size=(64, 64), stride=1).cuda(cuda_devices[i])
                                              for i in range(nstack)])

    def forward(self, imgs):
        '''
        Constructing the Stacked Hourglass Posenet Model
        :param imgs:
        :return:
        '''
        # x is of shape: (BatchSize, #channels == 3, input_dim1, input_dim2)
        x = imgs.permute(0, 3, 1, 2).cuda(self.cuda_devices[0])
        x = self.pre(x)
        combined_hm_preds = []
        hourglass_dict= {}

        for i in range(self.nstack):
            hourglass_dict = self.hgs[i](x.cuda(self.cuda_devices[i]))
            hg = hourglass_dict['out']

            # Hourglass parameters
            hourglass_dict[5] = self.hg_avg_pool[i](hg).clone().detach().reshape(hg.shape[0], -1)
            hourglass_dict['feature_5'] = hg.clone().detach()
            feature = self.features[i](hg)

            hourglass_dict['penultimate'] = self.global_avg_pool[i](feature).clone().detach().reshape(feature.shape[0], -1)

            preds = self.outs[i](feature)
            combined_hm_preds.append(preds.cuda(self.cuda_devices[-1]))
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        # ip_learn_loss_dict is a dictionary containing GAP outputs of hourglass
        return torch.stack(combined_hm_preds, 1), hourglass_dict