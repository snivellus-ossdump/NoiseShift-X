"""Small residual denoiser (DnCNN-style): the backbone shared by EVERY model in the project.

The network predicts the noise in the image; the clean estimate is input minus predicted noise
(residual learning, Zhang et al., IEEE TIP 2017). 10 conv layers, 64 channels, about 0.3 M parameters.
"""
import torch, torch.nn as nn

class SmallDenoiser(nn.Module):
    def __init__(self, ch=64, depth=10):
        super().__init__()
        # First layer: 3 (RGB) input channels -> 64 feature maps, followed by ReLU.
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
        # Middle layers (depth - 2 = 8): 3x3 conv, batch normalisation, ReLU.
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        # Last layer maps back to 3 channels: this is the predicted NOISE, not the clean image.
        layers += [nn.Conv2d(ch, 3, 3, padding=1)]
        # Plain stack of the 10 conv layers (receptive field 21 x 21 pixels).
        self.net = nn.Sequential(*layers)

    def forward(self, x):          # x in [0,1]; predicts the noise residual
        # Residual learning: denoised image = noisy input - predicted noise.
        return x - self.net(x)
