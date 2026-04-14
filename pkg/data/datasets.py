import torch
import pandas as pd
import nibabel as nib
import numpy as np
import os
import random 

from torch.utils.data import Dataset, Subset

from pkg.utils.diagnosis_matching import match_diagnosis
from pkg.utils.multimodality import create_multimodal_dataframe

from tqdm import tqdm

class ADNIDataset(Dataset):
    
    def __init__(self,
                 data_dir=None,
                 scan_csv=None, 
                 cached_samples=None,
                 modalities={
                     "MRI":["MRI-T1-3T"],
                     "PET":["PET-FDG"] }, 
                 diagnosis=[1,2,3], 
                 tolerance=180,
                 verbose=2):
    
        self.data_dir = data_dir
        self.scan_csv = scan_csv
        self.tolerance = tolerance
        self.cached_samples = cached_samples
        self.modalities = modalities 
        self.diagnosis = sorted(diagnosis)
        self.verbose = verbose
        
    def setup(self):

        if self.cached_samples is not None:

            # Load multimodal samples from cache 
            self.df_multimodal = pd.read_csv(self.cached_samples)

            # Get diagnoses 
            self.diagnosis = sorted(list(self.df_multimodal["diagnosis"].unique()))

            # Get modalities names
            modalities = []
            for col in self.df_multimodal.columns: 
                if self.df_multimodal[col].apply( lambda x : isinstance(x, str) and x.startswith("I")).any(): 
                    modalities.append(col)

            # If modalities are not specified, construct
            if self.modalities is None:
                self.modalities = {m:"" for m in modalities}

            # Construct scan df 
            series = []

            for mode in modalities:
                df = self.df_multimodal[ self.df_multimodal[mode].notna() ][mode]
                series.append(df)

            self.df_scan = pd.DataFrame( {"image_id":pd.concat(series)})
            self.df_scan = self.df_scan.reset_index(drop=True)



        else:

            # Load scans dataframe
            df_scan = pd.read_csv(self.scan_csv)

            # Filter for those available in the data dir
            available_scans = os.listdir(self.data_dir)
            df_scan = df_scan[ df_scan["image_id"].isin(available_scans)]

            # Create modalities from groups as described in the parameter "modalities"
            df_scan["modality"] = ""

            for mode, groups in self.modalities.items():
                df_scan.loc[df_scan["group"].isin(groups), "modality"] = mode 

            # Filter for specified modalities 
            df_scan = df_scan[ df_scan["modality"].isin(list(self.modalities.keys()))].copy()

            # Filter for specified diagnosis
            self.df_scan = df_scan[ df_scan["diagnosis"].isin(self.diagnosis)].copy()

        if self.verbose > 0:
                print("Verifying scans available in dir...")

        # Add file paths to df_scan
        paths = []

        for index, row in self.df_scan.iterrows():

            id = row["image_id"]
            allowed_filenames = ['clean_w_masked_m' + id + '.npy', 
                                'clean_w_masked_rstatic_' + id + '.npy']
            
            found = False
            for fname in allowed_filenames:

                path = os.path.join( self.data_dir, id, fname )
                if os.path.exists(path):
                    paths.append(path)
                    found = True
                    break

            if not found:
                paths.append(None)

        self.df_scan["path"] = paths
        self.df_scan = self.df_scan[ self.df_scan["path"].notna()]

        if self.cached_samples is None:

            if self.verbose > 0:
                print("Creating multimodal samples...")

            # Z-score normalization for numerical variables 
            self.df_scan["MMSE"] = (self.df_scan["MMSE"] - self.df_scan["MMSE"].mean())/self.df_scan["MMSE"].std()
            self.df_scan["CDR"] = (self.df_scan["CDR"] - self.df_scan["CDR"].mean())/self.df_scan["CDR"].std()
            self.df_scan["age"] = (self.df_scan["age"] - self.df_scan["age"].mean())/self.df_scan["age"].std()

            # Set gender to {0,1}
            self.df_scan["gender"] -= 1

            # Create multimodal samples
            self.df_multimodal = create_multimodal_dataframe(self.df_scan, tolerance=self.tolerance)

        self.df_multimodal = self.df_multimodal.reset_index(drop=True)
        self.df_scan = self.df_scan.reset_index(drop=True)

        # Add labels: map diagnosis to 0, ... , |classes|
        diag_to_label = {diag: i for i, diag in enumerate(self.diagnosis)}
        self.df_multimodal['label'] = self.df_multimodal['diagnosis'].map(diag_to_label)

        # Prepare samples now to speed up __getitem__ 
        self._prepare_samples()

        # Set all active modalities by default
        self.set_active_modalities("all")

    def _prepare_samples(self):

        self.modalities_sorted = sorted(self.modalities.keys())
        self.rows = self.df_multimodal.reset_index(drop=True)

        # Fast lookup: image_id -> npy path
        self.path_by_image_id = dict(zip(self.df_scan["image_id"], self.df_scan["path"]))

        self.n_modalities = len(self.modalities_sorted)
        self.scan_shape = (1, 91, 109, 91)

        self.samples = []

        for _, row in self.rows.iterrows():
      
            modes = {}
            for mode in self.modalities_sorted:
                image_id = row[mode]

                if isinstance(image_id, str):

                    modes[mode] = {"path":self.path_by_image_id[image_id], "mask":1.0}
                
                else:
                    modes[mode] = {"path":None, "mask":0.0}

            self.samples.append({
                "modes": modes,
                "label": int(row["label"]),
                "subject": row["subject_id"],
                "age": float(row["age"]) if "age" in row and pd.notna(row["age"]) else -512.0,
                "gender": float(row["gender"]) - 1 if "gender" in row and pd.notna(row["gender"]) else -512.0,
                "mmse": float(row["MMSE"]) if "MMSE" in row and pd.notna(row["MMSE"]) else -512.0,
                "cdr": float(row["CDR"]) if "CDR" in row and pd.notna(row["CDR"]) else -512.0,
                "key": row["strat_key"],
            })

    def set_active_modalities(self, active_modalities="all"):

        if active_modalities == "all":
            self.active_modalities = sorted(self.modalities.keys())
        else:
            if not isinstance(active_modalities, list):
                raise ValueError("Active modalities must be 'all' or a list")
            
            if not set(active_modalities) <= set(self.modalities.keys()):
                raise ValueError("Invalid modality name in active modalities")

            self.active_modalities = sorted(active_modalities)

    def __len__(self):
        return len(self.df_multimodal)
    
    def __getitem__(self, index):

        sample = self.samples[index]

        X = torch.zeros((len(self.active_modalities), *self.scan_shape), dtype=torch.float32)
        mask = torch.tensor( [sample["modes"][mode]["mask"] for mode in self.active_modalities], dtype=torch.float32)

        for i, mode in enumerate(self.active_modalities): 
            path = sample["modes"][mode]["path"]

            if path is None:
                continue 

            vol = np.load(path, mmap_mode="r")   # shape: (91, 109, 91) or similar
            X[i, 0] = torch.from_numpy(np.asarray(vol, dtype=np.float32))

        y = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "X": X,
            "y": y,
            "mask": mask,
            "subject": sample["subject"],
            "age": torch.tensor(sample["age"], dtype=torch.float32),
            "gender": torch.tensor(sample["gender"], dtype=torch.float32),
            "mmse": torch.tensor(sample["mmse"], dtype=torch.float32),
            "cdr": torch.tensor(sample["cdr"], dtype=torch.float32),
            "key": sample["key"],
        }

    def groups(self):
        return self.df_multimodal["subject_id"].astype(str).tolist()

    def labels(self):
        return self.df_multimodal["label"].astype(int).tolist()

    def strat_keys(self):
        return self.df_multimodal["strat_key"].astype(str).tolist()


class ADNITestDataset(ADNIDataset):

    def __init__(self,
                 data_dir,
                 scan_csv, 
                 cached_samples=None,
                 modalities={
                     "MRI":["MRI-T1-3T"],
                     "PET":["PET-FDG"] }, 
                 diagnosis=[1,2,3], 
                 tolerance=180,
                 verbose=2,
                 size=None,
                 complete_only=False):
    
        self.data_dir = data_dir
        self.scan_csv = scan_csv
        self.modalities = modalities 
        self.diagnosis = diagnosis
        self.tolerance = tolerance
        self.verbose = verbose
        self.cached_samples = cached_samples

        self.size = size
        self.complete_only = complete_only

    def setup(self):
        super().setup()

        if self.complete_only:

            modes = list(self.modalities.keys())
            mask = self.df_multimodal[modes].notna().all(axis=1)
            self.df_multimodal = self.df_multimodal[mask]
            self.df_multimodal = self.df_multimodal.reset_index(drop=True)

        if self.size is not None:

            # Take random subset of specified size 
            indices = [i for i in range(len(self.df_multimodal))]
            random.shuffle(indices)
            indices = indices[:self.size]
            self.df_multimodal = self.df_multimodal.iloc[ indices, : ]
            self.df_multimodal = self.df_multimodal.reset_index(drop=True)


class TransformDataset(Dataset):
    """
    Simple class that applies a transform to a dataset
    """
    def __init__(self, base_ds, transform=None):
        
        self.base = base_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):

        sample = self.base[idx]

        # Get tensor 
        X = sample["X"]

        # If it's single modality, apply transform directly 
        if len(X.shape) == 4: # (C,W,D,H)
            X = self.transform(X)

        elif len(X.shape) == 5: # (M,C,W,D,H)
            n_modalities = X.shape[0]

            # Temporarily flatten M and C dimensions, apply transform, then unflatten back
            X = self.transform(X.flatten(0,1)).unflatten(0, (n_modalities, -1))

        else:
            raise ValueError("Invalid sample shape")
            
        sample["X"] = X
        return sample
        
class DummyDataset(Dataset):

    def __init__(self, *args, **kwargs):
        pass 

    def setup(self):
        pass 

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return 1,1
 