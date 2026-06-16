import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset


def build_sampler(*, labels=None, weights=None):
    """
    Returns a WeightedRandomSampler.
    Provide either:
      - weights: array-like, len = num_samples
      - labels: labels of any hashable type, e.g. ints or strings
    """
    if weights is None:
        if labels is None:
            raise ValueError("build_sampler requires either weights or labels.")

        labels = np.asarray(labels)

        # Convert arbitrary labels/strings to integer group ids
        _, group_ids = np.unique(labels, return_inverse=True)

        group_counts = np.bincount(group_ids)
        group_weights = 1.0 / group_counts
        weights = group_weights[group_ids]

    w = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)



def build_loader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    weighted_sampling=False,
    labels=None,
    prefetch_factor=None
):
    """
    Simple loader builder.
    - Train: shuffle=True unless weighted_sampling=True (then sampler + shuffle=False)
    - Val/Test: shuffle=False, no sampler
    """
    sampler = None

    if weighted_sampling:
        if labels is None:
            raise ValueError("Labels are required for weighted sampling")
        sampler = build_sampler(labels=labels)
        shuffle = False  # cannot shuffle when sampler is used

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )
