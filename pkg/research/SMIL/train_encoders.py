
from pathlib import Path
import sys

# ==== Adjust path to make pkg imports work ===== #
ROOT = Path.cwd()
while not (ROOT / "pkg").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
print(ROOT)
# =============================================== #


from pkg.training.optimizer import build_optimizer
from pkg.training.criterion import build_criterion
from pkg.utils.instantiate import instantiate

import os
import sys
import yaml
import torch
import argparse

def train_encoders(config_path, save_dir): 

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.load(f, yaml.Loader)

    modalities = sorted(cfg["datamodule"]["args"]["data"]["modalities"].keys())

    # Setup data module
    dm = instantiate(cfg["datamodule"])
    dm.setup()

    epochs = int(cfg["trainer"]["max_epochs"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for mode in modalities: 

        print(f"Training encoder for {mode}")

        # Configure active modality in data modules 
        dm.set_active_modalities([mode])

        # Instantiate model, optimizer and criterion 
        model = instantiate(cfg["model"]).to(device)
        optim = build_optimizer(cfg['optimizer'], model.parameters())
        criterion = build_criterion(cfg['criterion'], train_labels=dm.train_labels, device=device)

        # Keep track of best val_loss and best model state
        best_val_loss = float('inf')
        best_model_state = None

        # Loop
        for epoch in range(epochs): 

            # Train
            train_loader = dm.train_dataloader(epoch)
            running_train_loss = 0.0 
            model.train()

            for batch in train_loader: 

                # Transfer batch to device 
                X = batch['X'].to(device)
                y = batch['y'].to(device)

                # Squeeze modality dimension
                X = X.squeeze(1)

                # Zero grad
                optim.zero_grad(set_to_none=True)

                # Forward pass
                loss = criterion( model(X), y)
                running_train_loss += loss.item()

                # Backward pass 
                loss.backward() 

                # Optimizer step 
                optim.step() 

            train_loss = running_train_loss/len(train_loader)

            # Validation
            model.eval()
            val_loader = dm.val_dataloader()
            running_val_loss = 0.0 

            with torch.no_grad():
                for batch in val_loader: 

                    # Transfer batch to device 
                    X = batch['X'].to(device)
                    y = batch['y'].to(device)

                    # Squeeze modality dimension
                    X = X.squeeze(1)

                    # Forward pass
                    loss = criterion( model(X), y)

                    running_val_loss += loss.item()

            val_loss = running_val_loss/len(val_loader)

            if val_loss <= best_val_loss: 
                best_val_loss = val_loss 
                best_model_state = model.state_dict()

                # Save to file 
                with open( os.path.join(save_dir, "encoder-" + mode + ".ckpt"), "wb") as f:
                    torch.save(best_model_state,f)

            print(f"Epoch {epoch+1} | Train loss: {train_loss} , Val loss: {val_loss}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train modality encoders for SMIL')
    
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="YAML config file"
    )

    parser.add_argument(
        '--save-dir',
        default=".",
        help='Directory where to save model dicts'
    )

    args = parser.parse_args()

    train_encoders(args.config, args.save_dir)