import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn(self.conv1(x)))
        # out = self.activation(self.conv1(x))
        return out


def _make_nConv(in_channel, out_channel, act):
    layer1 = LUConv(in_channel, out_channel, act)
    layer2 = LUConv(out_channel, out_channel, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, out_channel, act="relu", maxpool_ker=3, maxpool_stride=2):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, out_channel, act)
        self.maxpool = nn.MaxPool3d(maxpool_ker, stride=maxpool_stride)

    def forward(self, x):
        out_before_pool = self.ops(x)
        out = self.maxpool(out_before_pool)
        # print(out.size())
        return out


class DownTransition_mc(nn.Module):
    def __init__(self, in_channel, out_channel, act="relu", maxpool_ker=2, maxpool_stride=2):
        super(DownTransition_mc, self).__init__()
        self.ops = _make_nConv(in_channel, out_channel, act)
        self.mc = multi_crop(maxpool_ker=maxpool_ker, maxpool_stride=maxpool_stride)

    def forward(self, x):
        out_before_pool = self.ops(x)
        out = self.mc(out_before_pool)
        return out


class multi_crop(nn.Module):
    def __init__(self, maxpool_ker, maxpool_stride):
        super(multi_crop, self).__init__()
        # pooling层无参数，定义一个或多个pooling层无影响
        self.maxpool1 = nn.MaxPool3d(maxpool_ker, stride=maxpool_stride)
        self.maxpool2 = nn.MaxPool3d(maxpool_ker, stride=maxpool_stride)
        self.maxpool3 = nn.MaxPool3d(maxpool_ker, stride=maxpool_stride)

    def forward(self, x):
        # print(x.size())
        r0 = x
        r1 = r0[:, :, int(len(r0[1]) / 4):int(len(r0[1]) / 4 * 3),
             int(len(r0[1]) / 4):int(len(r0[1]) / 4 * 3),
             int(len(r0[1]) / 4):int(len(r0[1]) / 4 * 3)]
        r2 = r1[:, :, int(len(r1[1]) / 4):int(len(r1[1]) / 4 * 3),
             int(len(r1[1]) / 4):int(len(r1[1]) / 4 * 3),
             int(len(r1[1]) / 4):int(len(r1[1]) / 4 * 3)]

        f0 = self.maxpool1(self.maxpool2(r0))
        f1 = self.maxpool3(r1)
        f2 = r2

        out = torch.cat((f0, f1), dim=1)
        # print(out.size())
        # print(f2.size())
        out = torch.cat((out, f2), dim=1)
        # print(out.size())
        return out


class MC_CNN(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, act='relu'):
        super(MC_CNN, self).__init__()

        self.down_tr32 = DownTransition_mc(1, 64)
        self.down_tr16 = DownTransition(192, 64)
        self.down_tr4 = DownTransition(64, 64, maxpool_ker=4, maxpool_stride=4)
        self.fc = FC(64)

    def forward(self, x):
        # print(x.shape)
        out = self.down_tr32(x)
        # print(out.shape)
        out = self.down_tr16(out)
        out = self.down_tr4(out)
        out = self.fc(out)

        return out


class FC(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channel):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 32, bias=True),
            # nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2, bias=True)
        )

    def forward(self, x):
        # print(x.size())
        self.out_avg = F.avg_pool3d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)
        # print(self.out_avg.shape)
        self.out = self.fc(self.out_avg)

        return self.out


if __name__ == "__main__":
    target_model = MC_CNN().cuda()
    summary(target_model, (1, 64, 64, 64))
