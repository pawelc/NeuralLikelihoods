import SharedArray as sa
import os
import tempfile
import time
import numpy as np

class SharedMemory:

    def __init__(self, id = None):
        self._tmp_file = None
        if id is None:
            self._tmp_file = tempfile.mkstemp("RM_ipc")[1]
            self._id = os.path.basename(self._tmp_file)
        else:
            self._id = id
        self._step = 1
        self._status_name = "status"
        self._status = None
        self._names = set()

    @property
    def id(self):
        return self._id

    def read(self, names):
        if self._status is None:
            self._status = self.get_shm(self._status_name, wait=True)

        while self._status[0] != self._step:
            time.sleep(1)

        self._step+=1

        return {name:self.get_shm(name) for name in names}

    def write(self, values_dict):
        for name,value in values_dict.items():
            shm = sa.create("shm://{}_{}".format(self._id, name), value.shape, dtype=np.float32)
            shm[:] = value

        if self._status is None:
            self._status = sa.create("shm://{}_{}".format(self._id, self._status_name), 1, dtype=np.int32)

        self._status[0] = self._step
        self._step += 1

        self._names.update([name for name in values_dict.keys()])

    def get_shm(self, name, wait=True):
        done = False
        while not done:
            try:
                shm = sa.attach("shm://{}_{}".format(self._id, name))
                done = True
            except FileNotFoundError:
                if wait:
                    time.sleep(1)
                else:
                    return None
        self._names.add(name)
        return shm

    def delete_shm(self, names):
        for name in names:
            try:
                sa.delete("shm://{}_{}".format(self._id, name))
            except:
                pass

    def close(self):
        if self._tmp_file is not None:
            try:
                os.remove(self._tmp_file)
            except:
                pass

        self.delete_shm(self._names)


