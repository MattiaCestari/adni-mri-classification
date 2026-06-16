import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    """https://arxiv.org/abs/1710.05941"""
    return x * F.sigmoid(x)

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)

class SAM3d(nn.Module):
    """
    3d Spatial attention module
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv3d(2,1,kernel_size=7,padding=7//2)
        
    def forward(self, x):
        
        channel_max = torch.max(x, 1, keepdim=True)[0]
        channel_avg = torch.sum(x, 1, keepdim=True)/x.shape[1]
        
        m = torch.concat((channel_avg, channel_max), dim=1)
        m = torch.sigmoid(self.conv(m))
        
        return m*x
    

class ResidualBlock(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, use_sam=True, use_swish=True):
        super().__init__()

        self.use_swish = use_swish

        self.residual = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel_size, stride=stride, padding=kernel_size//2),
            nn.GroupNorm(8, channel_out),
            Swish() if use_swish else nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, channel_out),
            SAM3d() if use_sam else nn.Identity(),
        )
        
        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=stride, padding=0)
        
    def forward(self, x):
        
        x = self.residual(x) + self.skip1(x)
        x = swish(x) if self.use_swish else F.relu(x)
        return x


class Expert(nn.Module):

    def __init__(self, latent_dim, base=16, use_swish=True):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(1, base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, base),
            Swish() if use_swish else nn.ReLU(inplace=True),

            ResidualBlock(base, base, use_swish=use_swish),

            ResidualBlock(base, base*2, stride=2, use_swish=use_swish),
            ResidualBlock(base*2, base*2, use_swish=use_swish),

            ResidualBlock(base*2, base*4, stride=2, use_swish=use_swish),
            ResidualBlock(base*4, base*4, use_swish=use_swish),

            #SelfAttention3D(base*4, base), 

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        self.mlp_mu = nn.Linear(base*4, latent_dim)
        self.mlp_logvar = nn.Linear(base*4, latent_dim) 

    def forward(self, x):

        x = self.net(x)
        mu = self.mlp_mu(x)
        logvar = self.mlp_logvar(x)

        return mu, logvar