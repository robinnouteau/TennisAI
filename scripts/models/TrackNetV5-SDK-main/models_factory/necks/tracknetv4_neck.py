import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import NECKS
from ..basic.conv_block import BasicConvBlock as ConvBlock

@NECKS.register_module
class TrackNetV4Neck(nn.Module):
    def __init__(self, in_channels_list=[512, 256, 128, 64], out_channels=64):
        super().__init__()
        
        # --- Paramètres apprenants (La touche "V4") ---
        # Ces deux poids permettent au modèle d'ajuster la sensibilité au mouvement
        self.alpha = nn.Parameter(torch.tensor(1.0)) # Poids du mouvement
        self.beta = nn.Parameter(torch.tensor(0.0))  # Biais du mouvement

        # --- Décodeur Classique ---
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion1 = nn.Sequential(
            ConvBlock(in_channels_list[0] + in_channels_list[1], 256),
            ConvBlock(256, 256)
        )

        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion2 = nn.Sequential(
            ConvBlock(256 + in_channels_list[2], 128),
            ConvBlock(128, 128)
        )

        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion3 = nn.Sequential(
            ConvBlock(128 + in_channels_list[3], 64),
            ConvBlock(64, out_channels)
        )

    def forward(self, features):
        """
        Input: dictionnaire 'features'
        features['attention']: [B, 2, H, W] -> La différence de frames du backbone
        """
        # 1. Reconstruction spatiale (U-Net)
        s1, s2, s3 = features['skip1'], features['skip2'], features['skip3']
        x = features['bottleneck']

        x = self.fusion1(torch.cat([self.ups1(x), s3], dim=1))
        x = self.fusion2(torch.cat([self.ups2(x), s2], dim=1))
        x = self.fusion3(torch.cat([self.ups3(x), s1], dim=1))

        # 2. Raffinement par Motion Attention (Logique V4 Apprenante)
        # On récupère la différence de frames calculée par le backbone
        motion_diff = features['attention'] 
        
        # On calcule le masque d'attention avec les paramètres alpha et beta
        # On utilise la moyenne sur les 2 différences de frames pour stabiliser
        motion_mask = torch.sigmoid(self.alpha * motion_diff.abs().mean(dim=1, keepdim=True) + self.beta)
        
        # 3. Fusion finale : on "sculpte" les features avec le masque
        # out = Features visuelles * Masque de mouvement
        out = x * motion_mask

        return out