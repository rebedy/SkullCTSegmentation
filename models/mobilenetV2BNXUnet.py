import os
import sys
import logging
import math
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from mobilenetV2BNX import MobileNetV2BNX, InvertedResidual

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


"""
No Batch Normalization MobilenetV2Unet
"""


class MobileNetV2BNXUnet(nn.Module):
    def __init__(self, n_channels=5, n_classes=3, pre_trained=None, mode='train'):
        super(MobileNetV2BNXUnet, self).__init__()

        self.mode = mode
        self.backbone = MobileNetV2BNX(n_channels=n_channels, n_classes=3)

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 3, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 3, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 3, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 3, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)
        # ###!###
        self.dconv5 = nn.ConvTranspose2d(16, 8, 4, padding=1, stride=2)
        # self.invres5 = InvertedResidual(   16, 8, 1, 6)  # inp, oup, stride, expand_ratio

        # self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_last = nn.Conv2d(8, n_classes, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)
        # self.conv_score = nn.Conv2d(n_channels, n_classes, 1)

        self._init_weights()

        if pre_trained is not None:
            checkpoint = torch.load(pre_trained)
            self.backbone.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, x):

        input_size = x.size()

        x = interpolate(x, size=[512, 512])

        for n in range(0, 2):
            x = self.backbone.features[n](x)

        x1 = x
        # print(x1.shape, 'x1')
        logging.debug((x1.shape, 'x1'))

        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        # print(x2.shape, 'x2')
        logging.debug((x2.shape, 'x2'))

        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        # print(x3.shape, 'x3')
        logging.debug((x3.shape, 'x3'))

        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        # print(x4.shape, 'x4')
        logging.debug((x4.shape, 'x4'))

        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        # print(x5.shape, 'x5')
        logging.debug((x5.shape, 'x5'))

        logging.debug((x4.shape, self.dconv1(x).shape, 'up1'))
        up1 = torch.cat([
            x4,
            self.dconv1(x, output_size=x4.size())
        ], dim=1)
        up1 = self.invres1(up1)
        # print(up1.shape, 'up1')
        logging.debug((up1.shape, 'up1'))

        up2 = torch.cat([
            x3,
            self.dconv2(up1, output_size=x3.size())
        ], dim=1)
        up2 = self.invres2(up2)
        # print(up2.shape, 'up2')
        logging.debug((up2.shape, 'up2'))

        up3 = torch.cat([
            x2,
            self.dconv3(up2, output_size=x2.size())
        ], dim=1)
        up3 = self.invres3(up3)
        # print(up3.shape, 'up3')
        logging.debug((up3.shape, 'up3'))

        up4 = torch.cat([
            x1,
            self.dconv4(up3, output_size=x1.size())
        ], dim=1)
        up4 = self.invres4(up4)
        # print(up4.shape, 'up4')
        logging.debug((up4.shape, 'up4'))

        up5 = self.dconv5(up4)
        # print(up5.shape, 'up5')
        logging.debug((up5.shape, 'up5'))

        x = self.conv_last(up5)
        # print(x.shape, 'conv_last')
        logging.debug((x.shape, 'conv_last'))

        # x = self.conv_score(x)
        # print(x.shape, 'conv_score')
        # logging.debug((x.shape, 'conv_score'))

        # if self.mode == "eval":
        #     x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #     # print(x.shape, 'interpolate')
        #     logging.debug((x.shape, 'interpolate'))

        x = interpolate(x, size=[input_size[2], input_size[3]])

        # x = torch.nn.Softmax(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2BNXUnet(pre_trained=None)
    net(torch.randn(1, 5, 800, 800))
