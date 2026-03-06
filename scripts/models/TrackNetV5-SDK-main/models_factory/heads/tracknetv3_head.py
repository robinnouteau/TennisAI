from ..builder import HEADS
import torch.nn as nn

@HEADS.register_module
class TrackNetV3Head(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.predictor = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.predictor(x))