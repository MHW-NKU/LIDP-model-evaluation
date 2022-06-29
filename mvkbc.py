import os

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import resnet50
import torchvision
import matplotlib.pyplot as plt
import time


def multi_view_patches_extraction(img_3d):
    # plt.figure(num=1, figsize=(8, 5), )
    # img_3d = img_3d.type(torch.cuda.FloatTensor)
    img_3d = img_3d.cpu().numpy()

    img_3d_shape = img_3d.shape
    view1 = img_3d[:, :, :, :, 31]
    view2 = img_3d[:, :, 31, :, :]
    view3 = img_3d[:, :, :, 31, :]

    view1 = torch.from_numpy(view1).cuda()
    view2 = torch.from_numpy(view2).cuda()
    view3 = torch.from_numpy(view3).cuda()

    view4 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view4[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view4[:, :, i, j] = img_3d[:, :, i, j, img_3d_shape[2] - 1 - i]

    view4 = torch.from_numpy(view4).cuda()
    view4 = view4.type(torch.cuda.FloatTensor)

    view5 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view5[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view5[:, :, i, j] = img_3d[:, :, img_3d_shape[2] - 1 - i, j, img_3d_shape[2] - 1 - i]

    view5 = torch.from_numpy(view5).cuda()
    view5 = view5.type(torch.cuda.FloatTensor)

    view6 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view6[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view6[:, :, i, j] = img_3d[:, :, j, img_3d_shape[2] - 1 - i, img_3d_shape[2] - 1 - i]

    view6 = torch.from_numpy(view6).cuda()
    view6 = view6.type(torch.cuda.FloatTensor)

    view7 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view7[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view7[:, :, i, j] = img_3d[:, :, j, i, img_3d_shape[2] - 1 - i]

    view7 = torch.from_numpy(view7).cuda()
    view7 = view7.type(torch.cuda.FloatTensor)

    view8 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view8[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view8[:, :, i, j] = img_3d[:, :, img_3d_shape[2] - 1 - i, i, j]

    view8 = torch.from_numpy(view8).cuda()
    view8 = view8.type(torch.cuda.FloatTensor)

    view9 = np.empty((img_3d_shape[0], img_3d_shape[1], img_3d_shape[2], img_3d_shape[2]))
    for i in range(img_3d_shape[2]):
        # view9[batch, channel].append([])
        for j in range(img_3d_shape[2]):
            view9[:, :, i, j] = img_3d[:, :, i, i, j]

    view9 = torch.from_numpy(view9).cuda()
    view9 = view9.type(torch.cuda.FloatTensor)

    return view1, view2, view3, view4, view5, view6, view7, view8, view9

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
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
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(out_chan)

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


def _make_nConv(in_channel, act):
    layer1 = LUConv(in_channel, in_channel, act)
    layer2 = LUConv(in_channel, in_channel, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, act="relu", maxpool_ker=3, maxpool_stride=2):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, act)
        self.maxpool = nn.MaxPool2d(maxpool_ker, stride=maxpool_stride)

    def forward(self, x):
        out_before_pool = self.ops(x)
        out = self.maxpool(out_before_pool)
        # print(out.size())
        return out


class KBC_submodel(nn.Module):
    def __init__(self, resnet50_1, resnet50_2, resnet50_3):
        super(KBC_submodel, self).__init__()
        self.resnet50_1 = resnet50_1
        self.resnet50_2 = resnet50_2
        self.resnet50_3 = resnet50_3

        self.fc = nn.Sequential(
            nn.Linear(6, 2, bias=True),
            nn.Softmax()
        )

    def forward(self, OA, HS, HVV):
        out1 = self.resnet50_1(HS)
        out2 = self.resnet50_2(OA)
        out3 = self.resnet50_3(HVV)

        out = torch.cat((out1, out2, out3), dim=1)

        out = self.fc(out)
        out = nn.functional.softmax(out, dim=1)

        return out


class mvkbc(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self):
        super(mvkbc, self).__init__()

        for i in range(9):
            for j in range(3):
                setattr(self, 'resnet50_' + str(i + 1) + '_' + str(j + 1), resnet50.resnet50_pretrain(pretrain=True))

        self.kbc1 = KBC_submodel(self.resnet50_1_1, self.resnet50_1_2, self.resnet50_1_3)
        self.kbc2 = KBC_submodel(self.resnet50_2_1, self.resnet50_2_2, self.resnet50_2_3)
        self.kbc3 = KBC_submodel(self.resnet50_3_1, self.resnet50_3_2, self.resnet50_3_3)
        self.kbc4 = KBC_submodel(self.resnet50_4_1, self.resnet50_4_2, self.resnet50_4_3)
        self.kbc5 = KBC_submodel(self.resnet50_5_1, self.resnet50_5_2, self.resnet50_5_3)
        self.kbc6 = KBC_submodel(self.resnet50_6_1, self.resnet50_6_2, self.resnet50_6_3)
        self.kbc7 = KBC_submodel(self.resnet50_7_1, self.resnet50_7_2, self.resnet50_7_3)
        self.kbc8 = KBC_submodel(self.resnet50_8_1, self.resnet50_8_2, self.resnet50_8_3)
        self.kbc9 = KBC_submodel(self.resnet50_9_1, self.resnet50_9_2, self.resnet50_9_3)

        self.fc = nn.Sequential(
            nn.Linear(18, 2, bias=True)
        )

    def forward(self, OA, HS, HVV):
        out1 = self.kbc1(OA[:, 0, ...], HS[:, 0, ...], HVV[:, 0, ...])
        out2 = self.kbc2(OA[:, 1, ...], HS[:, 1, ...], HVV[:, 1, ...])
        out3 = self.kbc3(OA[:, 2, ...], HS[:, 2, ...], HVV[:, 2, ...])
        out4 = self.kbc4(OA[:, 3, ...], HS[:, 3, ...], HVV[:, 3, ...])
        out5 = self.kbc5(OA[:, 4, ...], HS[:, 4, ...], HVV[:, 4, ...])
        out6 = self.kbc6(OA[:, 5, ...], HS[:, 5, ...], HVV[:, 5, ...])
        out7 = self.kbc7(OA[:, 6, ...], HS[:, 6, ...], HVV[:, 6, ...])
        out8 = self.kbc8(OA[:, 7, ...], HS[:, 7, ...], HVV[:, 7, ...])
        out9 = self.kbc9(OA[:, 8, ...], HS[:, 8, ...], HVV[:, 8, ...])

        out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8, out9), dim=1)

        out = torch.sigmoid(self.fc(out))

        return out


class FC(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_channel):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 2, bias=True)
        )

    def forward(self, x):
        self.out_avg = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)
        self.out = self.fc(self.out_avg)

        return self.out


if __name__ == "__main__":
    target_model = mvkbc().cuda()

    summary(target_model, ((9, 1, 64, 64), (9, 1, 64, 64)))
