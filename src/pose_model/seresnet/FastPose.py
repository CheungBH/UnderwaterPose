import torch
import torch.nn as nn
from torch.autograd import Variable

from .layers.SE_Resnet import SEResnet
from ..duc.DUC import DUC
from config.config import pose_cls, device
from config import config
from config.model_cfg import seresnet_cfg


class FastPose(nn.Module):
    DIM = 128

    def __init__(self, cfg_file):
        super(FastPose, self).__init__()

        cfg = None
        if cfg_file:
            with open(cfg_file) as file:
                data = file.readlines()
            cfg = data[0].replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").split(",")
            cfg = [int(i) for i in cfg]

        self.preact = SEResnet(cfg=cfg)

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        if "duc_se.pth" in config.pose_weight:
            self.conv_out = nn.Conv2d(self.DIM, 33, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_out = nn.Conv2d(
                self.DIM, pose_cls, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        # if "duc" in config.pose_weight:
        #     out = out.narrow(1, 0, body_part_num)

        return out


def createModel(cfg=None):
    if cfg is not None:
        cfg = seresnet_cfg[cfg]
    return FastPose(cfg)


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset, weight, cfg=None):
        super(InferenNet_fast, self).__init__()
        if device != "cpu":
            model = createModel(cfg=cfg).cuda()
        else:
            model = createModel(cfg=cfg)
        model.load_state_dict(torch.load(weight, map_location=device))

        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        if "duc" in config.pose_weight:
            out = out.narrow(1, 0, 17)

        return out


def test():
    net = createModel()
    y = net(torch.randn(1,3,64,64))
    print(net, file=open("FastPose.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()

