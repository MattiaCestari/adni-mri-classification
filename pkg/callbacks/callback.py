class Callback():

    def __init__(self):
        self.priority = 10 # Default priority

    def on_fit_start(self, context):
        pass

    def on_train_epoch_start(self, context):
        pass

    def on_train_batch_start(self, context):
        pass 

    def on_train_batch_end(self, context, out, batch, batch_idx):
        pass

    def on_train_epoch_end(self, context):
        pass 

    def on_fit_end(self, context):
        pass 

    def on_val_epoch_start(self, context):
        pass

    def on_val_batch_start(self, context):
        pass

    def on_val_batch_end(self, context, out, batch, batch_idx):
        pass 

    def on_val_epoch_end(self, context):
        pass

    def on_test_start(self, context):
        pass 

    def on_test_batch_start(self, context):
        pass 

    def on_test_batch_end(self, context, out, batch, batch_idx):
        pass 
    
    def on_test_end(self, context):
        pass

    def on_stage_change( self, context):
        pass 


