class Conf:
    # Configuration class

    def __init__(self):
        self.dir = None
        self.max_num_epochs = -1
        self.num_workers = None
        self.visible_device_list = None
        self.device_placement_mapping = []
        self.shuffle_train_data = True
        self._eval_batch_size = {"0":1024, "1":1024, "CPU": 1024}
        self.precision = "32"
        self.data_seed = 1
        self.print_progress = False
        self.data_subset = None

    @property
    def eval_batch_size(self):
        if self.visible_device_list:
            return self._eval_batch_size[str(self.visible_device_list[0])]
        else:
            return self._eval_batch_size["CPU"]

    @eval_batch_size.setter
    def eval_batch_size(self, size):
        self._eval_batch_size = size

    def values_affecting_experiment(self):
        return {"max_num_epochs": self.max_num_epochs}

    def __str__(self):
        return str(self.__dict__)

conf = Conf()