import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Union
from torch import Tensor


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
        out = self.channel_attention(x) * x
        # out = self.spatial_attention(out) * out
        return out
    

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
    """ Conv + ReLU + BN"""
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True, use_depthwise=False, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        if use_depthwise:
            self.conv = DepthwiseSeparableConv2D(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Double2DConv(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Double2DConv2(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Double2DConv2, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (1, 1), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

        self.conv_3 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_4 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

        self.conv_5 = Conv2DBlock(in_dim, out_dim, (5, 5), use_depthwise=use_depthwise)
        self.conv_6 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

        self.conv_7 = Conv2DBlock(out_dim*3, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)

        x2 = self.conv_3(x)
        x2 = self.conv_4(x2)

        x3 = self.conv_5(x)
        x3 = self.conv_6(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_7(x)
        x = x + x2

        return x


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim, use_depthwise=False):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3), use_depthwise=use_depthwise)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 1  # 4 for Resnet bu TrackNetv3 without expansion

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




# [3, 4, 6, 3]

DEFAULT_TRACKNET_V3 = ['2c2', '2c2', '2c2', '3c', '2c', '2c', '2c']

class TrackNetV3(nn.Module):
    """ Original structure but less two layers 
        Total params: 10,161,411
        Trainable params: 10,153,859
        Non-trainable params: 7,552
    """
    def make_conv(self, mode: str, in_dim, out_dim, use_depthwise):
        if str(mode) == '2c2':
            return Double2DConv2(in_dim=in_dim, out_dim=out_dim, use_depthwise=use_depthwise)
        elif str(mode) == '3c':
            return Triple2DConv(in_dim=in_dim, out_dim=out_dim, use_depthwise=use_depthwise)
        elif str(mode) == '2c':
            return Double2DConv(in_dim=in_dim, out_dim=out_dim, use_depthwise=use_depthwise)
        elif str(mode) == 'conv7':
            return nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=1, padding=3, bias=False)
        elif str(mode) == 'l3':
            return self._make_layer(Bottleneck, in_dim, out_dim, 3, 1, False)


    def _make_layer(self, 
        block: type[Union[BasicBlock, Bottleneck]],
        in_dim: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(64, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            Bottleneck(
                self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    groups=1,
                    base_width=64,
                    dilation=1,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def __init__(self, in_dim=9, out_dim=3, use_depthwise=False, channels=[64, 128, 256, 512],
                 convs=DEFAULT_TRACKNET_V3):
        self.num_frame = in_dim // 3
        super(TrackNetV3, self).__init__()
        self.inplanes = 64

        self.down_block_1 = self.make_conv(convs[0], in_dim=in_dim, out_dim=channels[0], use_depthwise=use_depthwise)
        self.down_block_2 = self.make_conv(convs[1], in_dim=channels[0], out_dim=channels[1], use_depthwise=use_depthwise)
        self.down_block_3 = self.make_conv(convs[2], in_dim=channels[1], out_dim=channels[2], use_depthwise=use_depthwise)
        self.bottleneck = self.make_conv(convs[3], in_dim=channels[2], out_dim=channels[3], use_depthwise=use_depthwise)
        self.up_block_1 = self.make_conv(convs[4], in_dim=channels[2] + channels[3], out_dim=channels[2], use_depthwise=use_depthwise)
        self.up_block_2 = self.make_conv(convs[5], in_dim=channels[1] + channels[2], out_dim=channels[1], use_depthwise=use_depthwise)
        self.up_block_3 = self.make_conv(convs[6], in_dim=channels[0] + channels[1], out_dim=channels[0], use_depthwise=use_depthwise)
        self.predictor = nn.Conv2d(channels[0], out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.cbam1 = CBAM(channel=channels[2])  # only channel attention
        self.cbam2 = CBAM(channel=channels[1])
        self.cbam3 = CBAM(channel=channels[0])

        self.cbam0_2 = CBAM(channel=channels[2])
        self.cbam1_2 = CBAM(channel=channels[1])
        self.cbam2_2 = CBAM(channel=channels[0])

        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2)


    def forward(self, x):
        """ model input shape: (F*3, 288, 512), output shape: (F, 288, 512) """
        x1 = self.down_block_1(x)                                   # -> (64, 288, 512)
        #print ('x1', x1.shape)
        x = self.maxpool(x1)                 # -> (64, 144, 256)
        #print ('x', x.shape)
        x2 = self.down_block_2(x)                                   # (128, 144, 256)
        #print ('x2', x2.shape)  # [1, 128, 144, 256]
        x = self.maxpool(x2)                 # (128, 72, 128)
        x3 = self.down_block_3(x)                                   # (256, 72, 128), one less conv layer

        x = self.maxpool(x3)                 # (256, 36, 64)
        x = self.bottleneck(x)                                      # (512, 36, 64)
        x3 = self.cbam0_2(x3)
        x = torch.cat([self.up(x), x3], dim=1)  # (768, 72, 128) 256+512
        
        x = self.up_block_1(x)                                      # (256, 72, 128), one less conv layer
        x = self.cbam1(x)
        x2 = self.cbam1_2(x2)
        x = torch.cat([self.up(x), x2], dim=1)  # (384, 144, 256) 256+128
        
        x = self.up_block_2(x)                                      # (128, 144, 256)
        x = self.cbam2(x)
        x1 = self.cbam2_2(x1)
        x = torch.cat([self.up(x), x1], dim=1)  # (192, 288, 512) 128+64
        
        x = self.up_block_3(x)                                      # (64, 288, 512)
        x = self.cbam3(x)
        x = self.predictor(x)                                       # (3, 288, 512)
        x = self.sigmoid(x)
        return  x



class BigTrackNetV3(nn.Module):
    """ Original structure but bigger 
    """
    def __init__(self, in_dim=9, out_dim=3):
        super(BigTrackNetV3, self).__init__()
        self.down_block_1 = Double2DConv2(in_dim=in_dim, out_dim=128)
        self.down_block_2 = Double2DConv2(in_dim=128, out_dim=256)
        self.down_block_3 = Double2DConv2(in_dim=256, out_dim=512)
        self.bottleneck = Triple2DConv(in_dim=512, out_dim=1024)
        self.up_block_1 = Double2DConv(in_dim=1536, out_dim=512)
        self.up_block_2 = Double2DConv(in_dim=768, out_dim=256)
        self.up_block_3 = Double2DConv(in_dim=384, out_dim=128)
        self.predictor = nn.Conv2d(128, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.cbam1 = CBAM(channel=512) #only channel attention
        self.cbam2 = CBAM(channel=256)
        self.cbam3 = CBAM(channel=128)

        self.cbam0_2 = CBAM(channel=512)
        self.cbam1_2 = CBAM(channel=256)
        self.cbam2_2 = CBAM(channel=128)

        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """ model input shape: (F*3, 288, 512), output shape: (F, 288, 512) """
        x1 = self.down_block_1(x)                                   # (64, 288, 512)
        x = self.maxpool(x1)                 # (64, 144, 256)
        x2 = self.down_block_2(x)                                   # (128, 144, 256)
        x = self.maxpool(x2)                 # (128, 72, 128)
        x3 = self.down_block_3(x)                                   # (256, 72, 128), one less conv layer
        x = self.maxpool(x3)                 # (256, 36, 64)
        x = self.bottleneck(x)                                      # (512, 36, 64)
        x3 = self.cbam0_2(x3)
        x = torch.cat([self.up(x), x3], dim=1)  # (768, 72, 128) 256+512
        
        x = self.up_block_1(x)                                      # (256, 72, 128), one less conv layer
        x = self.cbam1(x)
        x2 = self.cbam1_2(x2)
        x = torch.cat([self.up(x), x2], dim=1)  # (384, 144, 256) 256+128
        
        x = self.up_block_2(x)                                      # (128, 144, 256)
        x = self.cbam2(x)
        x1 = self.cbam2_2(x1)
        x = torch.cat([self.up(x), x1], dim=1)  # (192, 288, 512) 128+64
        
        x = self.up_block_3(x)                                      # (64, 288, 512)
        x = self.cbam3(x)
        x = self.predictor(x)                                       # (3, 288, 512)
        x = self.sigmoid(x)
        return  x

# from torchsummary import summary
# Tr = TrackNetV3()
# summary(Tr, (9, 288, 512))