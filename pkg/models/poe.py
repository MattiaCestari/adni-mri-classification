import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from .components import ResidualBlock
from itertools import combinations

# TODO: [OK] log unimodal losses and multimodal losses separately
# TODO: use weights for different losses (unimodal and multimodal), so that no loss dominates overall.
# TODO: [OK] test batch 
# TODO: [OK] confusion matrix based on stratification


class Expert(nn.Module):

    def __init__(self, latent_dim, base=16):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv3d(1, base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.ReLU(inplace=True),

            ResidualBlock(base, base),

            ResidualBlock(base, base*2, stride=2),
            ResidualBlock(base*2, base*2),

            ResidualBlock(base*2, base*4, stride=2),
            ResidualBlock(base*4, base*4),

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


class PoE(BaseModel):

    def __init__(self, n_classes, n_modalities, latent_dim=128, dropout=0):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.latent_dim = latent_dim

        self.experts = nn.ModuleList([  Expert(latent_dim) for i in range(n_modalities) ])

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def set_criterion(self, criterion):
        
        # Set reduction to none to compute masked loss
        criterion.reduction = "none"
        self.criterion = criterion
    
    def forward_experts(self, X):

        """
        Runs all experts on the batch. 
        """

        mus, logvars = [], []

        for i, expert in enumerate(self.experts):
            mu_i, logvar_i = expert(X[:, i])   # (B, L)
            mus.append(mu_i)
            logvars.append(logvar_i)
            
        mus = torch.stack(mus, dim=1)           # (B, M, L)
        logvars = torch.stack(logvars, dim=1)   # (B, M, L)

        return mus, logvars
    
    def forward_from_cache(self, mus, logvars, mask):
        """
        mus, logvars : (B, M, L)
        mask: (B, M)
        """
        mask = mask.to(mus.dtype).unsqueeze(-1)  # (B, M, 1)
        vars = torch.exp(logvars.clamp(-10, 10)) + 1e-8     # (B, M, L)
        T = (1/vars)    # (B, M, L)
        var = 1/( (T*mask).sum(dim=1) + 1)  # (B, L)
        mu = var*( T*mus*mask ).sum(dim=1)  # (B, L)

        return self.classifier(mu)  #(B, n_classes)
        
    def train_batch(self, batch, batch_index):

        X = batch["X"]
        y = batch["y"]
        mask = batch["mask"]  # (B, M), 0/1

        out = {}
        total_loss = 0.0
        n_losses = 0

        # Compute all experts once
        mus, logvars = self.forward_experts(X)  # (B, M, L)

        # Precompute combos once in __init__ ideally; shown inline here
        for r in range(1, self.n_modalities + 1):
            for combo in combinations(range(self.n_modalities), r):

                current = mask[:, combo].bool().all(dim=1)
                if not current.any():
                    continue

                idx = current.nonzero(as_tuple=True)[0]

                # Build combo mask for this subset
                combo_mask = torch.zeros((idx.numel(), self.n_modalities),
                                        device=mask.device, dtype=mask.dtype)
                combo_mask[:, list(combo)] = 1

                logits = self.forward_from_cache(mus[idx], logvars[idx], combo_mask)
                l = F.cross_entropy(logits, y[idx], weight=self.criterion.weight)

                out["loss_" + "".join(map(str, combo))] = l.item()
                total_loss = total_loss + l
                n_losses += 1

        # avoid div-by-zero
        if n_losses == 0:
            total_loss = torch.zeros((), device=y.device, requires_grad=True)
            out["loss"] = 0.0
            return total_loss, out

        loss = total_loss / n_losses
        out["loss"] = loss.item()
        
        return loss, out

    def validate_batch(self, batch, batch_index):
        loss, dic = self.train_batch(batch, batch_index)
        return dic

    @torch.no_grad()
    def test_batch(self, batch, batch_index):
        X = batch["X"]
        avail = batch["mask"]  # (B, M) 0/1
        y = batch["y"]

        mus, logvars = self.forward_experts(X)                 # (B, M, L)
        logits = self.forward_from_cache(mus, logvars, avail)  # uses all available modalities per sample

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1).values

        return {"targets": y, "preds": preds, "confs": confs}
