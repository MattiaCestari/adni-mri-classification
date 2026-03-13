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

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()

        self.k = k
        self.f = nn.Conv3d(in_channels, k, kernel_size=1)          # query
        self.g = nn.Conv3d(in_channels, k, kernel_size=1)          # key
        self.h = nn.Conv3d(in_channels, in_channels, kernel_size=1) # value
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        B, C, W, H, D = X.shape
        N = W * H * D

        queries = self.f(X).flatten(start_dim=2)   # (B, k, N)
        keys    = self.g(X).flatten(start_dim=2)   # (B, k, N)
        values  = self.h(X).flatten(start_dim=2)   # (B, C, N)

        attn = torch.matmul(queries.transpose(1, 2), keys) / (self.k ** 0.5)  # (B, N, N)
        attn = torch.softmax(attn, dim=-1)                   # normalize over keys

        out = torch.matmul(values, attn)                     # (B, C, N)
        out = out.view(B, C, W, H, D)
        out = self.v(out)

        return self.gamma * out + X
        

    
