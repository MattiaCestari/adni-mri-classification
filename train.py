
from pathlib import Path
import sys

# ==== Adjust path to make pkg imports work ===== #
ROOT = Path.cwd()
while not (ROOT / "pkg").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
print(ROOT)
# =============================================== #

import os
import sys
import yaml
import argparse
import subprocess
import shutil
import numpy as np
import random

from pkg.training.criterion import build_criterion
from pkg.utils.instantiate import instantiate
from pkg.training.trainer import Trainer
from datetime import datetime
from pathlib import Path

import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_reproducibility_info(config_file, exp_dir):

    # Seed everything
    seed = 42
    seed_everything(seed)
    
    os.makedirs(exp_dir, exist_ok=True)

    # Save seed
    with open(os.path.join(exp_dir, "seed.txt"), "w") as f:
        f.write(str(seed))

    os.makedirs(exp_dir, exist_ok=True)

    # Save copy of configuration file 
    shutil.copy(config_file, os.path.join(exp_dir, "config.yaml"))

    # Save commit hash
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )

    with open(os.path.join(exp_dir, "commit.txt"), "w") as f:
        f.write(commit.stdout)

    # Save git diff patch
    with open(os.path.join(exp_dir, "patch.diff"), "w") as f:
        subprocess.run(
            ["git", "diff", "HEAD"],
            stdout=f,
            check=True,
        )

    # Save python requirements
    with open(os.path.join(exp_dir, "requirements.txt"), "w") as f:
        subprocess.run(
            ["pip", "freeze"],
            stdout=f,
            check=True,
        )


def main(config_file, exp_dir):

    # Create experiment dir
    os.makedirs(exp_dir, exist_ok=True)

    # Dump reproducibility info
    save_reproducibility_info(config_file, exp_dir)

    # Read configuration file
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # Data module 
    dm = instantiate(cfg["datamodule"])

    # Setup 
    dm.setup()

    # Log datamodule info
    print(dm.info())

    # Dump data info for reproducibility
    dm.dump(exp_dir)

    # Loop over folds
    for fold in range(dm.n_folds()):

        # Create fold dir 
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        os.makedirs(fold_dir)

        # Set current fold
        dm.set_fold(fold)

        # Build model and criterion 
        model = instantiate(cfg["model"])
        criterion = build_criterion(cfg["criterion"], train_labels=dm.train_labels)

        # Set criterion on model
        model.set_criterion(criterion)

        # Callbacks 
        callbacks = instantiate(cfg["callbacks"])

        # Build trainer 
        trainer = Trainer(model, dm, callbacks, cfg["optimizer"], dir=fold_dir, **cfg["trainer"])
        
        # Fit model 
        trainer.fit()
        
        # Test best model (loads automatically)
        trainer.test()

def compute_run_dir(args):
    if args.run_dir is not None:
        return args.run_dir

    run_dir = os.path.join(args.exp_dir, args.name)
    if not args.no_timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(run_dir, stamp)
    return run_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train or submit as Slurm job')
    
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="YAML config file"
    )

    parser.add_argument(
        '--job',
        action='store_true',
        help='Submit via Slurm'
    )

    parser.add_argument(
        '--exp-dir',
        default="./experiments",
        help='Experiments directory'
    )

    parser.add_argument(
        '--name',
        default=None,
        help='Experiment name'
    )

    parser.add_argument(
        '--time',
        default="6:00:00",
        help='Time limit'
    )

    parser.add_argument(
        '--mem',
        default="4G",
        help='Memory'
    )

    parser.add_argument(
        '--cores',
        default=1,
        help='Cores'
    )

    parser.add_argument(
        "--no-timestamp", 
        action="store_true",
        help="Do not append timestamp subdir")

    parser.add_argument(
        "--run-dir", 
        default=None,
        help="Explicit run directory (overrides exp-dir/name/timestamp)"
    )
    
    args = parser.parse_args()

    # Create experiment directory 
    run_dir = compute_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)

    # If job, fill template and submit
    if args.job:
        template = Path("job_template.slurm").read_text()
        slurm_script = template.format(
            job_name=args.name,
            time=args.time,
            log_out=os.path.join(run_dir, "slurm.out"),
            log_err=os.path.join(run_dir, "slurm.err"),
            config=os.path.abspath(args.config),
            exp_dir=os.path.abspath(args.exp_dir),
            cores=args.cores,
            mem=args.mem,
            name=args.name,
            no_timestamp="--no-timestamp" if args.no_timestamp else "",
            run_dir=os.path.abspath(run_dir),
        )

        slurm_path = os.path.join(run_dir, "job.slurm")
        Path(slurm_path).write_text(slurm_script)

        subprocess.run(["sbatch", slurm_path], check=True)
        sys.exit(0)

    main(args.config, run_dir)