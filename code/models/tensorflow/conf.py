class TfConf:

    def __init__(self):
        self.save_checkpoints_steps = 1000
        self.save_checkpoints_epochs = None
        self.save_summary_steps = 500
        self.save_summary_epochs = None
        self.print_tensors = False
        self.print_tensors_filter = ""
        self.print_tensors_summarize = 1000000
        self.debug_tb = False
        self.debug_cli = False
        self.debug_grad = False
        self.summary = False
        self.summary_filter = ""
        self.check_nans = False
        self.eval_throttle_secs = 120
        self.start_delay_secs = 0
        self.per_process_gpu_memory_fraction = 0.02
        self.allow_growth = True
        self.log_device_placement = False
        self.start_eval_step = 0
        self.per_process_gpu_memory = {'0':1024}
        self.disable_opt = False

    def values_affecting_experiment(self):
        return {"save_checkpoints_steps":self.save_checkpoints_steps,
                "save_summary_steps":self.save_summary_steps,
                "eval_throttle_secs":self.eval_throttle_secs,
                "start_delay_secs":self.start_delay_secs}

    def __str__(self):
        return str(self.__dict__)


tf_conf = TfConf()