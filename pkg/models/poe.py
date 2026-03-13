import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from .components import ResidualBlock, Swish, SelfAttention3D
from itertools import combinations

# TODO: [OK] log unimodal losses and multimodal losses separately
# TODO: use weights for different losses (unimodal and multimodal), so that no loss dominates overall.
# TODO: [OK] test batch 
# TODO: [OK] confusion matrix based on stratification


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

            SelfAttention3D(base*4, base), 

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

    def __init__(self, n_classes, n_modalities, base=16, training_combos=None, latent_dim=128, dropout=0, staged_training=False):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.latent_dim = latent_dim
        self.staged_training = staged_training

        if self.staged_training:
            self.head_frozen = False

        self.experts = nn.ModuleList([  Expert(latent_dim, base=base) for i in range(n_modalities) ])

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

        # If training combinations are not specified, compute them
        if training_combos is None and not staged_training:
            self.training_combos = []
            for r in range(1, self.n_modalities + 1):
                for combo in combinations(range(self.n_modalities), r):
                    self.training_combos.append(list(combo))

        else:
            self.training_combos = training_combos

    def set_criterion(self, criterion):
        
        # Set reduction to none to compute masked loss
        criterion.reduction = "none"
        self.criterion = criterion
    
    def forward_experts(self, X, modes="all"):
        """
        Runs specified experts on the batch.

        Returns
        -------
        mus : (B, M, L)
        logvars : (B, M, L)

        Non-requested modalities are filled with zeros.
        """
        B = X.shape[0]
        device = X.device
        dtype = X.dtype

        # figure out latent dim once
        latent_dim = self.latent_dim   # or however you store it

        mus = torch.zeros(B, self.n_modalities, latent_dim, device=device, dtype=dtype)
        logvars = torch.zeros(B, self.n_modalities, latent_dim, device=device, dtype=dtype)

        if modes == "all":
            mode_indices = range(self.n_modalities)
        elif isinstance(modes, (tuple, list)):
            mode_indices = modes
        else:
            raise ValueError("modes must be 'all' or a tuple/list of modality indices")

        for i in mode_indices:
            mu_i, logvar_i = self.experts[i](X[:, i])   # (B, L)
            mus[:, i] = mu_i
            logvars[:, i] = logvar_i

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
    
    def train_batch(self, batch, batch_index, stage=None):

        X = batch["X"]
        y = batch["y"]
        mask = batch["mask"]  # (B, M), 0/1

        out = {}
        total_loss = 0.0
        n_losses = 0

        if self.staged_training:
            if stage == 0: 
                # Optimize first expert + head 
                self.training_combos = [ [0,] ]
                
            elif stage > 0 and stage <= self.n_modalities-1:
                
                # Freeze head
                if not self.head_frozen:
                    for param in self.classifier.parameters(): 
                        param.requires_grad = False
                    self.head_frozen = True

                self.training_combos = [ [stage,] ] # Optimize single experts 

            else: 
                # Unfreeze head
                if self.head_frozen: 
                    for param in self.classifier.parameters(): 
                        param.requires_grad = True
                    self.head_frozen = False

                # Optimize all experts together
                self.training_combos = [ [i for i in range(self.n_modalities)] ]

        # Compute required modalities (union of combos) to be forwarded once 
        required_modes = sorted(set(i for combo in self.training_combos for i in combo))

        # Compute required experts once
        mus, logvars = self.forward_experts(X, modes=required_modes)  # (B, M, L)

        for combo in self.training_combos:

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
            return total_loss, out

        loss = total_loss / n_losses
        out["loss"] = loss.item()
        
        return loss, out


    def validate_batch(self, batch, batch_index, stage=None):
        loss, dic = self.train_batch(batch, batch_index, stage=stage)
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
