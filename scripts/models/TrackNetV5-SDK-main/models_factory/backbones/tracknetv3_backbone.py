import torch
import torch.nn as nn
from ..builder import BACKBONES

from ..basic.tracknetv3_basics import Double2DConv2, Triple2DConv

@BACKBONES.register_module
class TrackNetV3Backbone(nn.Module):
    def __init__(self, in_dim=9, channels=[64, 128, 256, 512], use_depthwise=False):
        super().__init__()
        # On réutilise tes classes Double2DConv2 et Triple2DConv définies dans tracknetv3.py
        self.down_block_1 = Double2DConv2(in_dim=in_dim, out_dim=channels[0], use_depthwise=use_depthwise)
        self.down_block_2 = Double2DConv2(in_dim=channels[0], out_dim=channels[1], use_depthwise=use_depthwise)
        self.down_block_3 = Double2DConv2(in_dim=channels[1], out_dim=channels[2], use_depthwise=use_depthwise)
        self.bottleneck = Triple2DConv(in_dim=channels[2], out_dim=channels[3], use_depthwise=use_depthwise)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        features = {}
        x1 = self.down_block_1(x)
        features['skip1'] = x1
        
        x = self.maxpool(x1)
        x2 = self.down_block_2(x)
        features['skip2'] = x2
        
        x = self.maxpool(x2)
        x3 = self.down_block_3(x)
        features['skip3'] = x3
        
        x = self.maxpool(x3)
        features['bottleneck'] = self.bottleneck(x)
        return features