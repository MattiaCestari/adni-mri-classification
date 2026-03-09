from .callback import Callback
from pkg.utils.plot import plot_metrics

import os

class Metrics(Callback):
    """
    Collects and aggregates (averages) scalar metrics from training and validation batch steps. 
    """
    def __init__(self, priority=0):

        self.current_train_metrics = {}
        self.history_train_metrics = {}

        self.current_val_metrics = {}
        self.history_val_metrics = {}

        # Set lowest priority by default
        self.priority = priority

    def on_fit_start(self, context):
        
        # Initialize csv logging path
        self.log_csv = os.path.join(context["dir"], f"metrics_stage_{context['stage']}.csv")
        self.first_epoch = True

    def on_train_epoch_start(self, context):
        self.current_train_metrics = {}

    def on_train_batch_end(self, context, out, batch, batch_idx):
        for k, v in out.items():
            if hasattr(v, "detach"):
                v = v.detach()

            # 🔧 CHANGED: store list instead of sum
            self.current_train_metrics.setdefault(k, []).append(v)

    def on_train_epoch_end(self, context):
        for k, v_list in self.current_train_metrics.items():
            # 🔧 CHANGED: average over actual occurrences
            avg = sum(v_list) / len(v_list) if len(v_list) > 0 else 0
            context["metrics"]["train/" + k] = avg
            self.history_train_metrics.setdefault(k, []).append(avg)

    def on_val_epoch_start(self, context):
        self.current_val_metrics = {}

    def on_val_batch_end(self, context, out, batch, batch_idx):
        for k, v in out.items():
            if hasattr(v, "detach"):
                v = v.detach()

            # 🔧 CHANGED: store list instead of sum
            self.current_val_metrics.setdefault(k, []).append(v)

    def on_val_epoch_end(self, context):
        for k, v_list in self.current_val_metrics.items():
            # 🔧 CHANGED: average over actual occurrences
            avg = sum(v_list) / len(v_list) if len(v_list) > 0 else 0
            context["metrics"]["val/" + k] = avg
            self.history_val_metrics.setdefault(k, []).append(avg)

        # write CSV
        keys = [k for k, _ in sorted(context["metrics"].items())]
        vals = [context["metrics"][k] for k in keys]

        # convert tensors to scalars
        def to_scalar(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu()
                if x.numel() == 1:
                    return float(x)
            return x

        vals = [to_scalar(v) for v in vals]

        with open(self.log_csv, "a") as f:
            if self.first_epoch:
                self.first_epoch = False
                f.write(",".join(keys) + "\n")
            f.write(",".join(map(str, vals)) + "\n")
    
    def on_fit_end(self, context):
        all_metrics = { "train/"+k: v for k, v in self.history_train_metrics.items() }
        for k, v in self.history_val_metrics.items():
            all_metrics["val/" + k] = v
        plot_metrics(os.path.join(context["dir"], f"metrics_stage{context['stage']}.png"), all_metrics)

    def on_stage_change(self, context):

        if context["stage"] > 0: 
            # Plot metrics
            all_metrics = { "train/"+k: v for k, v in self.history_train_metrics.items() }
            for k, v in self.history_val_metrics.items():
                all_metrics["val/" + k] = v
            plot_metrics(os.path.join(context["dir"], f"metrics_stage{context['stage']-1}.png"), all_metrics)

        # Update log csv 
        self.log_csv = os.path.join(context["dir"], f"metrics_stage{context['stage']}.csv")
        self.first_epoch = True

        # Reset metrics
        self.current_train_metrics = {}
        self.history_train_metrics = {}

        self.current_val_metrics = {}
        self.history_val_metrics = {}

        context["metrics"] = {}


