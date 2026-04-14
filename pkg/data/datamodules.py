from .datasets import ADNIDataset, TransformDataset, DummyDataset, ADNITestDataset
from .splitter import Splitter
from torch.utils.data import DataLoader, Subset
from .augmentation import build_augmentation
from .loaders import build_loader

import numpy as np 
import pandas as pd
import os
import json


class DataModule:
    """
    Skeleton for DataModule class.
    """
    
    def __init__(self ):
        raise NotImplementedError

    def setup(self, dataset, split, loader, transform):
        raise NotImplementedError

    def set_fold(idx):
        raise NotImplementedError

    def n_folds():
        raise NotImplementedError
    
    def dump(self, path):    # Reproducibility
        raise NotImplementedError

    def load(self, path):    # Reproducibility
        raise NotImplementedError
    
    def train_dataloader(self,epoch):
        raise NotImplementedError 

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


class ADNIDataModule(DataModule):
    """
    Data module for the ADNI dataset. Returns train, validation and test loaders,
    given cross validation splitting, dataset, loader and augmentation transform configs. 
    """

    def __init__(self, data, split, loader, transform):

        self.data_cfg = data
        self.split_cfg = split
        self.loader_cfg = loader
        self.transform_cfg = transform

        self.fold_index = 0

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        # Create dataset
        self.ds = ADNIDataset(**self.data_cfg)
        
    def setup(self):

        #Setup dataset
        self.ds.setup()

        # Create splitter
        splitter = Splitter(range(len(self.ds)), self.ds.labels(), self.ds.groups(),**self.split_cfg)

        # Save test indices
        self._test_idx = splitter.test_split()

        # Get train and validation indices for each fold
        self.folds = splitter.cv_split()

        # Set all modalities active by default
        self.set_active_modalities("all")

    def set_active_modalities(self, active_modalities="all"):

        # Configure dataset
        self.ds.set_active_modalities(active_modalities)
        active_modalities = self.ds.active_modalities
        
        # Only use indices where at least one of the selected modalities are available
        df = self.ds.df_multimodal[ self.ds.df_multimodal[ active_modalities ].notna().any(axis=1) ].copy()
        self.avail_idx = set(list(df.index))

        # Create test dataset and loader using intersections between split indices and available indices 
        avail_test_indices = list(set(self._test_idx).intersection(self.avail_idx))
        test_dataset = Subset(self.ds, avail_test_indices)

        loader_cfg = self.loader_cfg
        loader_cfg["shuffle"] = False
        loader_cfg["weighted_sampling"] = False
        self._test_loader = build_loader(test_dataset, labels=None, **loader_cfg)

        #print(f"{len(self.ds)} {len( avail_test_indices )} {len(test_dataset)} {len(self._test_loader)} ")

        # Set fold again if specified
        if self.fold_index is not None:
            self.set_fold(self.fold_index)

    def set_fold(self, idx):

        if idx > len(self.folds) - 1:
            raise IndexError(f"Fold index {idx} out of bounds for {len(self.folds)} folds")

        self.fold_index = idx

        # Indices   
        train_idxs = list(  set(self.folds[idx][0]).intersection(self.avail_idx) )
        val_idxs = list( set(self.folds[idx][1]).intersection(self.avail_idx) ) 

        # Augmentation transform
        augmentation = build_augmentation(self.transform_cfg)
        
        # Datasets
        train_ds = TransformDataset(Subset(self.ds, train_idxs), transform=augmentation)
        val_ds = Subset(self.ds, val_idxs)

        # Store train_labels (can be used for computing weights for loss criterion)
        self.train_labels = np.array(self.ds.labels())[train_idxs]

        # Loaders 
        self._train_loader = build_loader(train_ds, labels=self.train_labels, **self.loader_cfg)

        self.loader_cfg["shuffle"] = False  # Turn off for validation
        self.loader_cfg["weighted_sampling"] = False 
        self._val_loader = build_loader(val_ds, **self.loader_cfg)

    def n_folds(self):
        return len(self.folds)
    
    def train_dataloader(self, epoch):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def dump(self, dir):

        # Dump df_scan and df_multimodal from dataset
        self.ds.df_multimodal.to_csv(os.path.join(dir, "samples.csv"))

        indices = {"test_idx": [int(e) for e in self._test_idx] }

        for i, fold in enumerate(self.folds):
            train_idxs = [int(e) for e in fold[0]]
            val_idxs = [int(e) for e in fold[1]]

            indices[f"fold_{i}"] = {"train_idx":train_idxs, "val_idx":val_idxs}

        # Dump indices
        with open(os.path.join(dir, "indices.json"), "w") as f:
            f.write(json.dumps(indices))

    def info(self):

        # Unique stratification keys
        unique_keys = self.ds.df_multimodal["strat_key"].unique().tolist()

        # Row and column multi indices for info dataframe
        row_index = pd.MultiIndex.from_product([ 
                        [f"Fold {i}" for i in range(len(self.folds)) ], 
                        ["Train", "Val"]]
                    )
        col_index = pd.MultiIndex.append(
            pd.MultiIndex.from_tuples( [ ("#", ""), ("%", "")]),
            pd.MultiIndex.from_product([ ["Stratification Keys Distribution"], unique_keys])
        )

        # Create empty dataframe
        df = pd.DataFrame(index=row_index, columns=col_index)

        for i, fold in enumerate(self.folds):

            train_idxs, val_idxs = fold
            tot = len(train_idxs) + len(val_idxs)

            # Sample counts
            df.loc[(f"Fold {i}", "Train"), ("#","")] = len(train_idxs)
            df.loc[(f"Fold {i}", "Val"), ("#","")] = len(val_idxs)

            # As percentage
            df.loc[(f"Fold {i}", "Train"), ("%","")] = float(len(train_idxs)/tot)
            df.loc[(f"Fold {i}", "Val"), ("%","")] = float(len(val_idxs)/tot)

            # Stratification keys distribution
            train_keys_dist = self.ds.df_multimodal.loc[train_idxs, :].groupby("strat_key")["diagnosis"].count()/len(train_idxs)
            val_keys_dist = self.ds.df_multimodal.loc[val_idxs, :].groupby("strat_key")["diagnosis"].count()/len(val_idxs)

            for key in train_keys_dist.index:
                df.loc[(f"Fold {i}", "Train"), ("Stratification Keys Distribution",key)] = train_keys_dist[key]

            for key in val_keys_dist.index:
                df.loc[(f"Fold {i}", "Val"), ("Stratification Keys Distribution", key)] = val_keys_dist[key]

        # Add test dataset statistics 
        df.loc[("Test",""),("#","")] = len(self._test_idx)
        df.loc[("Test",""),("%","")] =  float(1)

        test_keys_dist = self.ds.df_multimodal.loc[self._test_idx, :].groupby("strat_key")["diagnosis"].count()/len(self._test_idx)
        
        for key in test_keys_dist.index:
            df.loc[("Test",""),("Stratification Keys Distribution", key)] = test_keys_dist[key]

        # Round numbers 
        df = df.apply(pd.to_numeric)
        df = df.round(3)
        return df

class ADNITestDataModule(ADNIDataModule):
    """
    ADNI Datamodule with customizable size for quick testing. 
    """
    
    def __init__(self, data, split, loader, transform, size=None, complete_only=False):
        self.data_cfg = data
        self.split_cfg = split
        self.loader_cfg = loader
        self.transform_cfg = transform

        self.fold_index = 0

        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        # Create dataset
        self.ds = ADNITestDataset(size=size,complete_only=complete_only, **self.data_cfg)


class DummyDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        self.ds = DummyDataset()
        self._train_loader = DataLoader(self.ds)
        self._val_loader = DataLoader(self.ds)
        self._test_loader = DataLoader(self.ds)
        self.train_labels = [1,1,1,1,1,1,1,1,1,1]

    def setup(self):
        pass 

    def set_fold(self, fold):
        pass 

    def n_folds(self):
        return 5
    
    def train_dataloader(self, epoch):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def info(self):
        return "Dummy Dataset"


    