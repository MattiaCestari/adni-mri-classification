import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .components import SAM3d

class BasicBlock1(nn.Module):
    
    def __init__(self, channel_in):
        super().__init__()
        
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.GroupNorm(8, channel_in),
            nn.ReLU(),
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.GroupNorm(8, channel_in),
            SAM3d()
        )
        
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.GroupNorm(8, channel_in),
            nn.ReLU(),
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.GroupNorm(8, channel_in),
            SAM3d()
        )
        
    def forward(self, x):
        
        x = F.relu(self.residual1(x) + x)
        x = F.relu(self.residual2(x) + x)
        
        return x

class BasicBlock2(nn.Module):
    
    def __init__(self, channel_in):
        super().__init__()
        
        channel_out = 2*channel_in
        
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, 3, stride=2, padding=3//2),
            nn.GroupNorm(8, channel_out),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.GroupNorm(8, channel_out),
            SAM3d()
        )
        
        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=2, padding=0)
        
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.GroupNorm(8, channel_out),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.GroupNorm(8, channel_out),
            SAM3d()
        )
        
    def forward(self, x):
        
        x = F.relu(self.residual1(x) + self.skip1(x))
        x = F.relu(self.residual2(x) + x)
        
        return x
    
    
class ResNet18(BaseModel):
    
    def __init__(self, num_classes=3, initial_filters=8, dropout=0):
        
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, initial_filters, 7, padding=7//2),
            nn.GroupNorm(8, initial_filters),
            nn.ReLU()
        )
        
        self.pool1 = nn.MaxPool3d(2)
        
        self.bb_1 = BasicBlock1(initial_filters)
        self.bb_2 = BasicBlock2(initial_filters)
        self.bb_3 = BasicBlock2(initial_filters*2)
        self.bb_4 = BasicBlock2(initial_filters*4)

        self.norm = nn.GroupNorm(8, initial_filters*8)
        self.avgpool = nn.AvgPool3d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1728, num_classes)
        
    def forward(self, x):
        
        x = self.pool1(self.initial_block(x))
        
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        
        x = self.avgpool(self.norm(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        return self.fc1(x)

    def train_batch(self, batch, batch_idx):
        X,y = batch
        out = self.forward(X)
        loss = self.criterion(out, y)
        return loss, {"loss":float(loss.item())}

    def validate_batch(self, batch, batch_idx):
        X,y = batch
        out = self.forward(X)
        loss = self.criterion(out, y)
        return {"loss":float(loss.item())}

    def test_batch(self, batch, batch_idx):
        X,y = batch
        y_hat = self.forward(X).argmax(dim=1)
        return {"preds":y_hat, "targets":y}
