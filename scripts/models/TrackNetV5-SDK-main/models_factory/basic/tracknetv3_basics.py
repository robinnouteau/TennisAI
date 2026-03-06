import torch
import torch.nn as nn
from torch import Tensor

# --- MODULES D'ATTENTION (CBAM) ---

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        # Dans TrackNetV3, seule l'attention de canal est utilisée par défaut
        out = self.channel_attention(x) * x
        return out

# --- CONVOLUTIONS SPÉCIALISÉES ---

class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Conv2DBlock(nn.Module):
    """ Bloc Conv + BN + ReLU standard """
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True, use_depthwise=False):
        super(Conv2DBlock, self).__init__()
        if padding == 'same':
            # Calcul automatique du padding pour garder la taille identique
            pad = kernel_size[0] // 2 if isinstance(kernel_size, tuple) else kernel_size // 2
        else:
            pad = padding

        if use_depthwise:
            self.conv = DepthwiseSeparableConv2D(in_dim, out_dim, kernel_size=kernel_size[0], padding=pad, bias=bias)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=pad, bias=bias)
        
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# --- BLOCS DE RÉSEAU (TRACKNET V3) ---

class Double2DConv(nn.Module):
    """ Bloc décodeur : deux convolutions 3x3 successives """
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))

class Double2DConv2(nn.Module):
    """ Bloc multi-échelle (Inception-like) utilisé dans l'encodeur V3 """
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Double2DConv2, self).__init__()
        # Branche 1x1
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (1, 1), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        # Branche 3x3
        self.conv_3 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_4 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        # Branche 5x5
        self.conv_5 = Conv2DBlock(in_dim, out_dim, (5, 5), use_depthwise=use_depthwise)
        self.conv_6 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        # Fusion
        self.conv_7 = Conv2DBlock(out_dim * 3, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        x1 = self.conv_2(self.conv_1(x))
        x2 = self.conv_4(self.conv_3(x))
        x3 = self.conv_6(self.conv_5(x))
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.conv_7(out)
        return out + x2 # Connexion résiduelle avec la branche 3x3

class Triple2DConv(nn.Module):
    """ Bloc Bottleneck : trois convolutions 3x3 successives """
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))