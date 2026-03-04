import torch
import os
import pandas as pd
from .callback import Callback
from sklearn.metrics import classification_report, confusion_matrix

class TestReport(Callback):
    """
    Receives predictions and targets on test batches.
    Computes classification report, confusion matrix and other test metrics.
    """
    def __init__(self, priority=10):
        self.priority = priority

    def on_test_start(self, context):

        # Load best model state 
        context["model"].load_state_dict(torch.load(context["checkpoint_path"], weights_only=True))

        # Initialize predictions, targets, keys and confidences 
        self.preds = []
        self.targets = []
        self.keys = []
        self.confs = []

    def on_test_batch_end(self, context, out, batch, batch_idx):
        
        preds = out["preds"]
        targets = out["targets"]
        confs = out["confs"]

        if not isinstance(preds, torch.Tensor) or not isinstance(targets, torch.Tensor) or not isinstance(confs, torch.Tensor):
            raise TypeError("Expected preds/targets/confs to be torch.Tensor")

        preds = preds.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)
        confs = confs.detach().cpu().view(-1)

        self.preds.append(preds)
        self.targets.append(targets)
        self.confs.append(confs)
        self.keys += batch["key"]   # Append stratification keys
        
    def on_test_end(self, context):
        
        if len(self.preds) == 0 or len(self.targets) == 0:
            raise RuntimeError(f"No data collected: preds={len(self.preds)} targets={len(self.targets)}")

        # Compute global confusion matrix and classification report
        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()
        confs = torch.cat(self.confs).numpy()

        cm = pd.DataFrame(confusion_matrix(targets, preds))
        report = pd.DataFrame(classification_report(targets, preds, digits=4, output_dict=True)).T

        print(cm)
        print(report)

        # Save to .csv 
        cm.to_csv(os.path.join(context["dir"], "confusion_matrix.csv"))
        report.to_csv(os.path.join(context["dir"], "classification_report.csv"))

        # Dump dataframe with predictions, targets and stratification keys for 
        # more in-depth analysis
        pd.DataFrame({"pred":preds, "target":targets, "confidence":confs, "key":self.keys}).to_csv(
            os.path.join(context["dir"], "predictions.csv")
        )

        # reset for next run
        self.preds.clear()
        self.targets.clear()
        