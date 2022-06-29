import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SAM(nn.Module):
    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.in_ch = in_ch
        self.AVGPool = nn.AvgPool2d(kernel_size=5, stride=2, padding=0)
        self.fc = nn.Sequential(nn.Linear(36, 1))
        for i in range(in_ch):
            setattr(self, 'fc' + str(i + 1), self.fc)

    def forward(self, x):
        out = torch.empty((x.shape[0], 0)).cuda()

        for i in range(x.shape[1]):
            feature_map = x[:, i]
            feature_map = feature_map.unsqueeze(1)

            fc = getattr(self, 'fc' + str(i + 1))
            out_after_pool = self.AVGPool(feature_map)
            tmp = out_after_pool.view(out_after_pool.size()[0], -1)
            out = torch.cat([out, fc(tmp)], dim=1)

        return out


class HESAM(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, in_ch=11, out_ch=64):
        super(HESAM, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 256)

        self.up7 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.GMP = nn.AdaptiveMaxPool2d(1)

        self.basic_block1 = BasicBlock(64, 128,
                                       downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                                                nn.BatchNorm2d(128)))
        self.basic_block2 = BasicBlock(128, 256, stride=2,
                                       downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                                                                nn.BatchNorm2d(256)))

        self.SAM = SAM(256)

        self.FC = nn.Sequential(nn.Linear(256, 2))

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        gmp = self.GMP(c4)

        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)

        out = self.basic_block1(out)
        out = self.basic_block2(out)

        out = self.SAM(out)
        gmp = gmp.squeeze(dim=2)
        gmp = gmp.squeeze(dim=2)
        out = out + gmp
        out = self.FC(out)

        return out


if __name__ == "__main__":
    device = torch.device('cuda')
    target_model = HESAM()
    target_model = target_model.to(device)
    summary(target_model, (21, 32, 32))
