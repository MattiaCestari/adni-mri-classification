
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from .callback import Callback

SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": {"class": ReduceLROnPlateau, "step_on": "epoch", "monitor": "val/loss"},
    "CosineAnnealingLR": {"class": CosineAnnealingLR, "step_on": "epoch", "monitor": None},
    "OneCycleLR": {"class": OneCycleLR, "step_on": "batch", "monitor": None},
}

class SchedulerPolicy(Callback):
    """
    Implements LR scheduler policy, as well as stages for curriculum learning. 
    """
    def __init__(self, name, params, priority=10):
        
        if name not in SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler '{name}'. Options: {list(SCHEDULER_REGISTRY)}")

        # Save config and params for later initialization
        self.params = params
        self.conf = SCHEDULER_REGISTRY[name]

        self.priority = priority

    def on_fit_start(self, context):

        # Get optimizer from context and initialize scheduler
        optim = context["optimizer"]
        self.scheduler = self.conf["class"](optim, **self.params)
    
    def on_stage_change(self, context):

        # Rebuild scheduler at every new stage
        optim = context["optimizer"]
        self.scheduler = self.conf["class"](optim, **self.params)
    
    def on_train_batch_end(self, context, out, batch, batch_idx):
        if self.conf["step_on"] == "batch":
            self.scheduler.step()

    def on_val_epoch_end(self, context):

        metrics = context["metrics"]
        if self.conf["step_on"] != "epoch":
            return

        monitor_key = self.conf["monitor"]
        if monitor_key is None:
            self.scheduler.step()
        else:
            if metrics is None or monitor_key not in metrics:
                raise KeyError(f"Scheduler '{self.name}' requires metric '{monitor_key}' at epoch end.")
            self.scheduler.step(metrics[monitor_key])