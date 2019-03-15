import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.up(x)

class U_net(nn.Module):
    def __init__(self, sync_bn=True):
         super(U_net, self).__init__()

         if sync_bn:
             BatchNorm = SynchronizedBatchNorm2d
         else:
             BatchNorm = nn.BatchNorm2d

         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

         self.Conv1 = conv_block(ch_in=3, ch_out=64, BatchNorm=BatchNorm)
         self.Conv2 = conv_block(ch_in=64, ch_out=128, BatchNorm=BatchNorm)
         self.Conv3 = conv_block(ch_in=128, ch_out=256, BatchNorm=BatchNorm)
         self.Conv4 = conv_block(ch_in=256, ch_out=512, BatchNorm=BatchNorm)
         self.Conv5 = conv_block(ch_in=512, ch_out=1024, BatchNorm=BatchNorm)

         self.Up5 = up_conv(ch_in=1024, ch_out=512, BatchNorm=BatchNorm)
         self.up_conv5 = conv_block(ch_in=1024, ch_out=512, BatchNorm=BatchNorm)

         self.Up4 = up_conv(ch_in=512, ch_out=256, BatchNorm=BatchNorm)
         self.up_conv4 = conv_block(ch_in=512, ch_out=256, BatchNorm=BatchNorm)

         self.Up3 = up_conv(ch_in=256, ch_out=128, BatchNorm=BatchNorm)
         self.up_conv3 = conv_block(ch_in=256, ch_out=128, BatchNorm=BatchNorm)

         self.Up2 = up_conv(ch_in=128, ch_out=64, BatchNorm=BatchNorm)
         self.up_conv2 = conv_block(ch_in=128, ch_out=64, BatchNorm=BatchNorm)

         self.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # ---------------------------

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill(1)
                m.bias.data.zero_()



    # def get_1x_1r_params(self):
    #     modules = [self.Conv1, self.Conv2, self.Conv3, self.Conv4, self.Conv5,
    #                self.Up2, self.Up3, self.Up4, self.Up5,
    #                self.up_conv5, self.up_conv4, self.up_conv3, self.up_conv2]
    #     for i in range(len(modules)):
    #         for j in modules[i].parameters():
    #             if j.


