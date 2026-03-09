
import torch 
from tqdm import tqdm

from pkg.training.optimizer import build_optimizer

class CallbacksList:
    def __init__(self, callbacks):
        self.callbacks = callbacks or []

        # Sort callbacks by ascending priority, reflecting order of execution
        self.callbacks.sort(key= lambda c : int(c.priority))

    def _call(self, name, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, name, None)
            if fn is None:
                continue
            fn(*args, **kwargs)

    def on_fit_start(self, context):
        self._call("on_fit_start", context)

    def on_train_epoch_start(self, context):
        self._call("on_train_epoch_start", context)

    def on_train_batch_start(self, context):
        self._call("on_train_batch_start", context)

    def on_train_batch_end(self, context, out, batch,  batch_idx):
        self._call("on_train_batch_end", context, out, batch, batch_idx)

    def on_train_epoch_end(self, context):
        self._call("on_train_epoch_end", context)

    def on_fit_end(self, context):
        self._call("on_fit_end", context)

    def on_val_epoch_start(self, context):
        self._call("on_val_epoch_start", context)

    def on_val_batch_start(self, context):
        self._call("on_val_batch_start", context)

    def on_val_batch_end(self, context, out, batch,  batch_idx):
        self._call("on_val_batch_end", context, out, batch,  batch_idx)

    def on_val_epoch_end(self, context):
        self._call("on_val_epoch_end", context)

    def on_test_start(self, context):
        self._call("on_test_start", context) 

    def on_test_batch_start(self, context):
        self._call("on_test_batch_start", context)

    def on_test_batch_end(self, context, out, batch, batch_idx):
        self._call("on_test_batch_end", context, out, batch, batch_idx)
    
    def on_test_end(self, context):
        self._call("on_test_end", context)

    def on_stage_change(self, context):
        self._call("on_stage_change", context)


class Trainer:

    def __init__(self, model, datamodule, callbacks,  optimizer_cfg, 
                 max_epochs=100,  
                 dir=None,
                 automatic_optimization=True,
                 accum_steps=1,
                 stages=[],
                 reset_optim=True):

        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.accum_steps = accum_steps
        self.automatic_optimization = automatic_optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stages = stages
        self.reset_optim = reset_optim

        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create callbacks list
        self.cb = CallbacksList(callbacks)

        # Build optimizer 
        self.optimizer = build_optimizer(optimizer_cfg, self.model.parameters())

        # Create context
        self.ctx = { 
            "epoch":1,
            "stage":0,
            "model":self.model,
            "optimizer":self.optimizer,
            "metrics":{},
            "signals":{ 
                "early_stop":False
            },
            "dir":dir,
            "device":self.device, 
        }

    def fit(self):

        self.cb.on_fit_start(self.ctx)

        print("Stage 0")

        for epoch in range(1,self.max_epochs+1):

            # Stop if early stop is signalled
            if self.ctx["signals"]["early_stop"]:
                break 

            # Set epoch in context
            self.ctx["epoch"] = epoch
            
            # Stage update logic
            new_stage = self._get_stage(epoch)
            if new_stage != self.ctx["stage"]: # Transition between stages 
                if self.reset_optim:
                    self.ctx["optimizer"] = build_optimizer(
                                self.optimizer_cfg, 
                                model_params=self.ctx["model"].parameters()
                    )
                    self.optimizer = self.ctx["optimizer"]
                self.ctx["stage"] = new_stage
                self.cb.on_stage_change(self.ctx)
                print(f"Stage {new_stage}")
            
            print(f"Epoch {epoch}")

            # Train
            self.model.train()
            self.cb.on_train_epoch_start(self.ctx)
            self.train_epoch(epoch)
            self.cb.on_train_epoch_end(self.ctx)

            # Validation
            self.model.eval()
            with torch.no_grad():
                self.cb.on_val_epoch_start(self.ctx)
                self.validate_epoch(epoch)
                self.cb.on_val_epoch_end(self.ctx)
            
        self.cb.on_fit_end(self.ctx)

    def test(self):
        
        # Test
        self.model.eval()
        with torch.no_grad():

            self.cb.on_test_start(self.ctx)

            for i, batch in enumerate(self.datamodule.test_dataloader()):

                # Move tensors to device
                for k, v in batch.items():    
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                self.cb.on_test_batch_start(self.ctx)
                out = self.ctx["model"].test_batch(batch, i)
                self.cb.on_test_batch_end(self.ctx, out, batch, i)

            self.cb.on_test_end(self.ctx)

    def train_epoch(self, epoch):
        
        # Gradient accumulation steps
        k = self.accum_steps

        # Train loader
        train_loader = self.datamodule.train_dataloader(epoch)

        if self.automatic_optimization:
            self.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):

            # Move tensors to device
            for key, v in batch.items():    
                if isinstance(v, torch.Tensor):
                    batch[key] = v.to(self.device)
     
            self.cb.on_train_batch_start(self.ctx)
            loss, step_out = self.model.train_batch(batch, i, stage=self.ctx["stage"])

            if self.automatic_optimization:
                (loss/k).backward()

            self.cb.on_train_batch_end(self.ctx, step_out, batch, i)

            if self.automatic_optimization and (i+1)%k == 0:
                # Gradient clipping ?
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
        if self.automatic_optimization and (i+1)%k != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def validate_epoch(self, epoch):

        val_loader = self.datamodule.val_dataloader()

        for i, batch in enumerate(val_loader):

            # Move tensors to device
            for key, v in batch.items():    
                if isinstance(v, torch.Tensor):
                    batch[key] = v.to(self.device)

            self.cb.on_val_batch_start(self.ctx)
            step_out = self.model.validate_batch(batch, i, stage=self.ctx["stage"])
            self.cb.on_val_batch_end(self.ctx, step_out, batch, i)

    def _get_stage(self, epoch):
        """
        Returns stage index given current epoch
        """
        stage = 0

        for limit in self.stages: 
            if epoch > limit:
                stage += 1 

        return stage