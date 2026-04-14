
import torch
import torch.nn as nn

from pkg.models.components import ResidualBlock, Swish, SelfAttention3D

class Encoder(nn.Module):
    """
    Encoder for 3d volumes.
    """
    
    def __init__(self, latent_dim, num_classes=3, base=16, use_swish=True):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv3d(1, base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, base),
            Swish() if use_swish else nn.ReLU(inplace=True),

            ResidualBlock(base, base, use_swish=use_swish),

            ResidualBlock(base, base*2, stride=2, use_swish=use_swish),
            ResidualBlock(base*2, base*2, use_swish=use_swish),

            ResidualBlock(base*2, base*4, stride=2, use_swish=use_swish),
            ResidualBlock(base*4, base*4, use_swish=use_swish),

            SelfAttention3D(base*4, base), 

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        self.proj = nn.Linear(base*4, latent_dim)

        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        return self.classifier(z)
