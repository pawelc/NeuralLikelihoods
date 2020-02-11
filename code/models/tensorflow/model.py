from copy import copy

from data import DataLoader


class Context:
    def __init__(self, data_loader: DataLoader, kwargs):
        self._data_loader = data_loader
        self._kwargs = kwargs

    @property
    def data_loader(self) -> DataLoader:
        return self._data_loader

    @property
    def kwargs(self) :
        return self._kwargs


class Model:
    def __init__(self, **kwargs):
        kwargs = copy(kwargs)
        if "name" in kwargs:
            self._name = kwargs["name"]
            del kwargs["name"]
        else:
            self._name = self.__class__.__name__

        self.__dict__.update(kwargs)

    """Called before model __call__ function is invoked
    """
    def init(self, context: Context):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplemented

    def name(self):
        return self._name

    def can_compute_marginals(self):
        return False



