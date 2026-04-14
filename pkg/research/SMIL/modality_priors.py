# ==== Adjust path to make pkg imports work ===== #
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "pkg").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
print(ROOT)
# =============================================== #

import yaml
import torch
import numpy as np

from tqdm import tqdm
from pkg.utils.instantiate import instantiate
from sklearn.cluster import KMeans

# Load dataset config and set it up
with open("./encoders.yaml", "r") as f:
    cfg = yaml.load(f, yaml.Loader)

dm = instantiate(cfg["datamodule"])
dm.setup()
dm.set_active_modalities(["PET-fdg"])

encoder_pet = instantiate(cfg["model"])
with open("./encoder-PET-fdg.ckpt", "rb") as f:
    encoder_pet.load_state_dict( torch.load(f, map_location="cpu"))


encodings = []

encoder_pet.eval()
with torch.no_grad():
    for batch in tqdm(dm.train_dataloader(0)): 
        
        X = batch["X"].squeeze(1)
        encodings.append( encoder_pet.encode(X))

all_encodings = torch.cat(encodings, dim=0).unbind(0)

kmeans = KMeans(
    n_clusters=3*10,
    init="k-means++",
    verbose=1,
)

kmeans.fit(all_encodings)
priors = kmeans.cluster_centers_

with open("./priors.npy","wb") as f:
    np.save(f, priors)

