class EarlyStop:
    def __init__(self, monitor_every_epoch, patience):
        self.monitor_every_epoch = monitor_every_epoch
        self.patience = patience