import torch
import torch.nn as nn
from ..builder import NECKS
from ..basic.tracknetv3_basics import CBAM, DepthwiseSeparableConv2D, Double2DConv

@NECKS.register_module
class TrackNetV3Neck(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], use_depthwise=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        
        # Blocs de remontée
        self.up_block_1 = Double2DConv(in_dim=channels[2] + channels[3], out_dim=channels[2], use_depthwise=use_depthwise)
        self.up_block_2 = Double2DConv(in_dim=channels[1] + channels[2], out_dim=channels[1], use_depthwise=use_depthwise)
        self.up_block_3 = Double2DConv(in_dim=channels[0] + channels[1], out_dim=channels[0], use_depthwise=use_depthwise)
        
        # Modules d'attention CBAM
        self.cbam1, self.cbam2, self.cbam3 = CBAM(channels[2]), CBAM(channels[1]), CBAM(channels[0])
        self.cbam0_2, self.cbam1_2, self.cbam2_2 = CBAM(channels[2]), CBAM(channels[1]), CBAM(channels[0])

    def forward(self, features):
        x1, x2, x3 = features['skip1'], features['skip2'], features['skip3']
        x = features['bottleneck']

        # Étape 1
        x3_att = self.cbam0_2(x3)
        x = torch.cat([self.up(x), x3_att], dim=1)
        x = self.cbam1(self.up_block_1(x))
        
        # Étape 2
        x2_att = self.cbam1_2(x2)
        x = torch.cat([self.up(x), x2_att], dim=1)
        x = self.cbam2(self.up_block_2(x))
        
        # Étape 3
        x1_att = self.cbam2_2(x1)
        x = torch.cat([self.up(x), x1_att], dim=1)
        x = self.cbam3(self.up_block_3(x))
        
        return x