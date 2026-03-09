import torch
import os 

from .callback import Callback

class CheckpointManager(Callback):
    def __init__(self, monitor="val/loss", mode="min", patience=10, priority=10):
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.reset()

        self.priority = priority

    def reset(self):
        self.patience_counter = 0
        self.best_metric = None

    def _is_better(self, current, best):
        return (current < best) if self.mode == "min" else (current > best)

    def on_fit_start(self, context):

        # Create and set checkpoint save path
        context["checkpoint_path"] = os.path.join(context["dir"], f"model_stage_{context['stage']}.ckpt")
    
    def on_val_epoch_end(self, ctx):
        
        if self.monitor not in ctx["metrics"]:
            raise KeyError(f"CheckpointManager requires '{self.monitor}' in ctx.metrics")

        current = float(ctx["metrics"][self.monitor])

        if self.best_metric is None or self._is_better(current, self.best_metric):
            self.best_metric = current
            self.patience_counter = 0
            torch.save(ctx["model"].state_dict(), ctx["checkpoint_path"])
        else:
            self.patience_counter += 1

        if self.patience > 0 and self.patience_counter >= self.patience:
            ctx["signals"]["early_stop"] = True

    def on_stage_change(self, context):
        
        # Load best model 
        state_dict = torch.load(
            context["checkpoint_path"],
            map_location=context["device"],
            weights_only=False
        )
        context["model"].load_state_dict(state_dict) 

        # Update checkpoint path
        context["checkpoint_path"] = os.path.join(context["dir"], f"model_stage_{context['stage']}.ckpt")

        # Reset
        self.reset()

    def on_test_start(self, context):
        
        # Load best model 
        state_dict = torch.load(
            context["checkpoint_path"],
            map_location=context["device"],
            weights_only=False
        )
        context["model"].load_state_dict(state_dict)