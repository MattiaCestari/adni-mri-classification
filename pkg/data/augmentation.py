import torch
import torch.nn.functional as F

import torchio as tio


class Augmentation():
    
    def __init__(self, config):
        pass 
    
    def transform_sample(self, x):
        pass
    
    def transform_batch(self, batch):
        pass 
    
    
def random_crop(
    volume: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
) -> torch.Tensor:
    """
    Randomly crop a sub-volume of `volume` and resize it
    back to the original shape using trilinear interpolation.
    Input:  4D torch.Tensor (C, D, H, W) on any device.
    Output: same shape & same device.
    """
    C, D, H, W = volume.shape
    device = volume.device

    # pick random scale
    scale = torch.empty(1, device=device).uniform_(min_scale, max_scale).item()
    new_D = max(1, int(D * scale))
    new_H = max(1, int(H * scale))
    new_W = max(1, int(W * scale))

    # pick random corner
    z0 = torch.randint(0, D - new_D + 1, (), device=device).item()
    y0 = torch.randint(0, H - new_H + 1, (), device=device).item()
    x0 = torch.randint(0, W - new_W + 1, (), device=device).item()

    # crop
    cropped = volume[
        :,
        z0 : z0 + new_D,
        y0 : y0 + new_H,
        x0 : x0 + new_W,
    ]  # shape (C, new_D, new_H, new_W)

    # prepare for interpolation: add batch dim
    cropped = cropped.unsqueeze(0)  # (1, C, new_D, new_H, new_W)

    # resize back to (D, H, W) with trilinear
    resized = F.interpolate(
        cropped,
        size=(D, H, W),
        mode='trilinear',
        align_corners=False,
    )
    return resized.squeeze(0)  # (C, D, H, W) on original device

@torch.no_grad()
def _one_hot(y: torch.Tensor, num_classes: int, dtype=None):
    oh = F.one_hot(y, num_classes=num_classes)
    return oh.to(dtype if dtype is not None else torch.float32)

def cutmix_3d(
    x: torch.Tensor,           # (B, C, D, H, W)
    y: torch.Tensor,           # (B,) int64 class indices
    num_classes: int,
    alpha: float = 1.0,
    per_sample: bool = False,  # if True, different box & lambda per sample
    same_class: bool = False,  # NEW: intra-class CutMix with hard labels
):
    """
    Returns:
      x_mixed: (B, C, D, H, W)
      y_soft:  (B, num_classes) soft targets
      perm:    (B,) donor indices actually used (per-sample)
      lam_vec: (B,) effective lambda per sample (after exact box volume)
    Notes:
      - Keeps everything in torch (GPU-friendly).
      - If B < 2, no mixing is applied (identity).
      - If same_class=True, donors are sampled within the same class and labels stay HARD.
    """
    assert x.dim() == 5, "x should be (B,C,D,H,W)"
    B, C, D, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Edge case: nothing to mix
    if B < 2:
        y_soft = _one_hot(y, num_classes, dtype=dtype)
        lam_vec = torch.ones(B, device=device, dtype=dtype)
        perm = torch.arange(B, device=device)
        return x, y_soft, perm, lam_vec

    # --- pick donors (perm) ---
    if same_class:
        # donor per sample from the same class (fallback to self if no other)
        perm = torch.empty(B, dtype=torch.long, device=device)
        for i in range(B):
            same_mask = (y == y[i])
            idxs = torch.nonzero(same_mask, as_tuple=False).squeeze(1)
            # exclude self if possible
            if idxs.numel() > 1:
                # choose any index != i
                # sample until not i (cheap for small batches)
                j = idxs[torch.randint(0, idxs.numel(), (1,), device=device)]
                if j.item() == i:
                    # try again once; if still i, keep i
                    j = idxs[torch.randint(0, idxs.numel(), (1,), device=device)]
                perm[i] = j
            else:
                perm[i] = i  # no alternative in batch -> self (no-op)
    else:
        # standard CutMix: random permutation donors
        perm = torch.randperm(B, device=device)

    donor_x = x[perm]         # (B,C,D,H,W)
    donor_y = y[perm]         # (B,)

    beta = torch.distributions.Beta(alpha, alpha)
    x_mixed = x.clone()

    # ----- draw cuboid(s) & paste -----
    if not per_sample:
        # Single lambda & cuboid for the whole batch
        lam = beta.sample().to(device=device, dtype=dtype)  # scalar
        cut_rat = torch.pow(1.0 - lam, 1.0 / 3.0)           # volume -> edge ratio
        cut_d = max(int(D * cut_rat.item()), 1)
        cut_h = max(int(H * cut_rat.item()), 1)
        cut_w = max(int(W * cut_rat.item()), 1)

        cd = torch.randint(0, D, (1,), device=device).item()
        ch = torch.randint(0, H, (1,), device=device).item()
        cw = torch.randint(0, W, (1,), device=device).item()

        d1 = max(cd - cut_d // 2, 0); d2 = min(cd + cut_d // 2, D)
        h1 = max(ch - cut_h // 2, 0); h2 = min(ch + cut_h // 2, H)
        w1 = max(cw - cut_w // 2, 0); w2 = min(cw + cut_w // 2, W)

        if (d2 - d1) > 0 and (h2 - h1) > 0 and (w2 - w1) > 0:
            x_mixed[:, :, d1:d2, h1:h2, w1:w2] = donor_x[:, :, d1:d2, h1:h2, w1:w2]

        vol = (d2 - d1) * (h2 - h1) * (w2 - w1)
        lam_eff = 1.0 - vol / float(D * H * W)  # exact kept ratio from x
        lam_vec = torch.full((B,), lam_eff, device=device, dtype=dtype)

    else:
        # Different cuboid per sample (useful for small 3D batches)
        lam_vec = torch.empty(B, device=device, dtype=dtype)
        for b in range(B):
            lam_b = beta.sample().to(device=device, dtype=dtype)
            cut_rat = torch.pow(1.0 - lam_b, 1.0 / 3.0)
            cut_d = max(int(D * cut_rat.item()), 1)
            cut_h = max(int(H * cut_rat.item()), 1)
            cut_w = max(int(W * cut_rat.item()), 1)

            cd = torch.randint(0, D, (1,), device=device).item()
            ch = torch.randint(0, H, (1,), device=device).item()
            cw = torch.randint(0, W, (1,), device=device).item()

            d1 = max(cd - cut_d // 2, 0); d2 = min(cd + cut_d // 2, D)
            h1 = max(ch - cut_h // 2, 0); h2 = min(ch + cut_h // 2, H)
            w1 = max(cw - cut_w // 2, 0); w2 = min(cw + cut_w // 2, W)

            if (d2 - d1) > 0 and (h2 - h1) > 0 and (w2 - w1) > 0:
                x_mixed[b, :, d1:d2, h1:h2, w1:w2] = donor_x[b, :, d1:d2, h1:h2, w1:w2]

            vol = (d2 - d1) * (h2 - h1) * (w2 - w1)
            lam_vec[b] = 1.0 - vol / float(D * H * W)

    # ----- labels -----
    y1h = _one_hot(y,     num_classes, dtype=dtype)
    y2h = _one_hot(donor_y, num_classes, dtype=dtype)

    if same_class:
        # intra-class: keep HARD labels
        y_soft = y1h
    else:
        # standard CutMix: soft labels weighted by exact kept ratio
        lam_b = lam_vec.view(B, 1)
        y_soft = lam_b * y1h + (1.0 - lam_b) * y2h

    return x_mixed, y_soft, perm, lam_vec

def mixup_3d(
    x: torch.Tensor,           # (B, C, D, H, W)
    y: torch.Tensor,           # (B,) class indices (int64)
    num_classes: int,
    alpha: float = 0.4,        # smaller alpha keeps images closer to originals
    per_sample: bool = True,   # per-sample λ is typical for MixUp
):
    """
    Returns:
      x_mix:  (B, C, D, H, W)
      y_soft: (B, num_classes)
      perm:   (B,) indices used to shuffle the batch
      lam:    (B,) effective λ per sample
    """
    B = x.size(0)
    device, dtype = x.device, x.dtype

    if B < 2:
        return x, _one_hot(y, num_classes, dtype=dtype), torch.arange(B, device=device), torch.ones(B, device=device, dtype=dtype)

    perm = torch.randperm(B, device=device)
    x2, y2 = x[perm], y[perm]

    beta = torch.distributions.Beta(alpha, alpha)
    if per_sample:
        lam = beta.sample((B,)).to(device=device, dtype=dtype)  # (B,)
        lam_x = lam.view(B, 1, 1, 1, 1)
    else:
        lam = beta.sample().to(device=device, dtype=dtype)      # ()
        lam_x = lam.view(1, 1, 1, 1, 1).expand(B, -1, -1, -1, -1)

    x_mix = lam_x * x + (1.0 - lam_x) * x2

    y1h = _one_hot(y,  num_classes, dtype=dtype)
    y2h = _one_hot(y2, num_classes, dtype=dtype)
    y_soft = lam.view(B, 1) * y1h + (1.0 - lam.view(B, 1)) * y2h

    return x_mix, y_soft, perm, lam

def build_augmentation(cfg: list) -> tio.Compose:
    """
    Build a torchio Compose where each transform
    takes & returns a torch.Tensor.
    """
    transforms = []
    for d in cfg:
        name = list(d.keys())[0]
        params = d.get(name, {})
        
        if name == 'random_crop':
            transforms.append(
                tio.Lambda(
                    lambda x: random_crop(
                        x,
                        params.get('min_scale', 0.5),
                        params.get('max_scale', 1.0),
                    ),
                    p=params.get('p', 0.5),
                )
            )
            
        elif name == 'random_flip':
            transforms.append(
                tio.RandomFlip(axes=('LR',), p=params.get('p', 0.5))
            )

        elif name == 'random_blur':
            transforms.append(
                tio.RandomBlur(std=params.get('std',0.1), p=params.get('p', 0.5))
            )

        elif name == 'random_affine':
            transforms.append(
                tio.RandomAffine(
                    scales=params.get('scales',0),
                    degrees=params.get('degrees', 10),
                    translation=params.get('translation',10),
                    p=params.get('p', 0.5),
                )
            )
        
        elif name == 'random_noise':
            transforms.append(
                tio.RandomNoise(
                    std=params.get('std', 0.05),
                    p=params.get('p', 0.2)
                )
            )
            
        else:
            raise ValueError(f"Unknown transform: {name}")
        
    return tio.Compose(transforms)

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union

@torch.no_grad()
def _resolve_box(D:int, H:int, W:int,
                 size_frac: Union[float, Tuple[float,float,float]],
                 size_voxels: Optional[Tuple[int,int,int]]):
    if size_voxels is not None:
        sd, sh, sw = size_voxels
    else:
        if isinstance(size_frac, (int, float)):
            size_frac = (float(size_frac),) * 3
        sd = max(1, int(round(D * size_frac[0])))
        sh = max(1, int(round(H * size_frac[1])))
        sw = max(1, int(round(W * size_frac[2])))
    # make sizes odd so center-padding math is exact
    if sd % 2 == 0: sd += 1
    if sh % 2 == 0: sh += 1
    if sw % 2 == 0: sw += 1
    return sd, sh, sw

@torch.no_grad()
def _center_mask_fits(mask: torch.Tensor, sd:int, sh:int, sw:int) -> torch.Tensor:
    """
    mask: (B,1,D,H,W) bool, True=inside brain.
    Returns: (B,1,D,H,W) bool of valid centers where a full (sd,sh,sw) box fits.
    """
    pd, ph, pw = sd//2, sh//2, sw//2
    inv = (~mask).float()  # 1 outside, 0 inside
    # pad with 1s so windows near borders count as 'outside'
    inv_pad = F.pad(inv, (pw, pw, ph, ph, pd, pd), value=1.0)
    pooled = F.max_pool3d(inv_pad, kernel_size=(sd, sh, sw), stride=1, padding=0)
    return (pooled < 0.5)

@torch.no_grad()
def cutout_3d(
    x: torch.Tensor,                            # (B,C,D,H,W)
    holes: int = 1,
    size_frac: Union[float, Tuple[float,float,float]] = 0.05,
    size_voxels: Optional[Tuple[int,int,int]] = None,
    fill: str = "mean",                         # "mean" | "zero" | "noise" | "const"
    const_val: float = 0.0,
    brain_mask: Optional[torch.Tensor] = None,  # (B,1,D,H,W) bool; if None → auto
    restrict_to_brain: bool = True,
    return_mask: bool = True,
):
    """
    Batched 3D CutOut with robust brain masking.

    - If brain_mask is None, builds a foreground mask that tolerates zeros inside the brain:
      threshold at 10% of per-sample max |x|, then fills small holes via 3D pooling.
    - If restrict_to_brain=True, centers are chosen so the entire cuboid lies within brain.
    - Holes can be specified by fractional size (per-dimension) or absolute voxels.
    """
    assert x.dim() == 5, "x must be (B,C,D,H,W)"
    B, C, D, H, W = x.shape
    device, dtype = x.device, x.dtype

    sd, sh, sw = _resolve_box(D, H, W, size_frac, size_voxels)

    # --- sampling mask (True = foreground) ---
    if brain_mask is None:
        # robust auto-mask: channel-avg magnitude, per-sample threshold, fill gaps
        mag = x.abs().mean(dim=1, keepdim=True)                             # (B,1,D,H,W)
        thr = (0.10 * mag.flatten(1).amax(dim=1)).view(B, 1, 1, 1, 1)
        rough = mag > thr
        samp_mask = (F.max_pool3d(rough.float(), kernel_size=5, stride=1, padding=2) > 0)
    else:
        samp_mask = brain_mask.bool()

    center_mask = _center_mask_fits(samp_mask, sd, sh, sw) if restrict_to_brain \
                  else torch.ones_like(samp_mask, dtype=torch.bool)

    x_out = x.clone()
    M = torch.zeros(B, 1, D, H, W, device=device, dtype=torch.bool)

    # --- fill stats ---
    if fill == "mean":
        mean_ch = x.mean(dim=(2,3,4), keepdim=True)                         # (B,C,1,1,1)
    elif fill == "noise":
        mean_ch = x.mean(dim=(2,3,4), keepdim=True)
        std_ch  = x.std(dim=(2,3,4), keepdim=True).clamp_min(torch.finfo(dtype).eps)
    elif fill == "const":
        const = torch.tensor(const_val, device=device, dtype=dtype)
    elif fill != "zero":
        raise ValueError("fill must be 'mean' | 'zero' | 'noise' | 'const'")

    # --- per-sample holes ---
    for b in range(B):
        centers = torch.nonzero(center_mask[b, 0], as_tuple=False)  # (N,3)
        for _ in range(holes):
            if centers.numel() == 0:
                # fallback anywhere in volume
                cd = torch.randint(0, D, (), device=device).item()
                ch = torch.randint(0, H, (), device=device).item()
                cw = torch.randint(0, W, (), device=device).item()
            else:
                idx = torch.randint(centers.shape[0], (), device=device).item()  # scalar index
                cd, ch, cw = centers[idx].tolist()  # three ints

            d1 = cd - sd//2; d2 = d1 + sd
            h1 = ch - sh//2; h2 = h1 + sh
            w1 = cw - sw//2; w2 = w1 + sw
            d1 = max(d1, 0); h1 = max(h1, 0); w1 = max(w1, 0)
            d2 = min(d2, D); h2 = min(h2, H); w2 = min(w2, W)
            if d1 >= d2 or h1 >= h2 or w1 >= w2:
                continue

            M[b, 0, d1:d2, h1:h2, w1:w2] = True

        # apply fill
        mask_bc = M[b].expand(C, D, H, W)  # broadcast to channels
        if fill == "zero":
            x_out[b].masked_fill_(mask_bc, 0)
        elif fill == "mean":
            x_out[b] = torch.where(mask_bc, mean_ch[b].expand_as(x_out[b]), x_out[b])
        elif fill == "noise":
            noise = torch.randn_like(x_out[b]) * std_ch[b] + mean_ch[b]
            x_out[b] = torch.where(mask_bc, noise, x_out[b])
        elif fill == "const":
            x_out[b].masked_fill_(mask_bc, const)

    return (x_out, M) if return_mask else x_out


if __name__ == "__main__":
    
    cfg = {
        'random_crop':{
            'min_scale':0.3,
            'max_scale':1.2,
            'order':1,
            'p':0.2,
        }
    }
    
    transform = build_augmentation(cfg)