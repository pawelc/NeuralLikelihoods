import concurrent.futures
import time
import traceback
from concurrent.futures import Executor

from conf import conf
from experiment.progress import ProgressMonitor, NoOpProgressMonitor


class SameProcessFuture:
    def __init__(self, res):
        self.res = res

    def result(self, timeout=None):
        return self.res


class SameProcessExecutor(Executor):

    def submit(self, fn, *args, **kwargs):
        return SameProcessFuture(fn(*args, **kwargs))


class WorkItem:
    def __init__(self, objective_fun, args_list, args_named, future):
        self.args_list = args_list
        self.args_named = args_named
        self.future = future
        self.objective_fun = objective_fun
        self.name = objective_fun.name(args_named)

class Callable:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*self.args, **self.kwargs)
        except:
            raise ValueError(traceback.format_exc())

def invoke_in_process_pool(name, num_workers, *funcs):
    if conf.print_progress:
        progress_mon = ProgressMonitor(1, name)
    else:
        progress_mon = NoOpProgressMonitor()

    done = 0.0
    futures = []
    res = [None] * len(funcs)
    with SameProcessExecutor() if num_workers <= 0 else concurrent.futures.ProcessPoolExecutor(
            num_workers) as executor:
        for i, fun in enumerate(funcs):
            inserted = False
            while not inserted:
                if len(futures) < num_workers or num_workers <= 0:
                    futures.append((i, executor.submit(fun)))
                    inserted = True

                for fut in list(futures):
                    try:
                        res[fut[0]] = fut[1].result(0)
                        done += 1
                        progress_mon.progress(done / len(funcs))
                        futures.remove(fut)
                    except concurrent.futures.TimeoutError:
                        pass

                if len(futures) == num_workers and num_workers > 0:
                    time.sleep(1)

    for fut in list(futures):
        res[fut[0]] = fut[1].result()
        done += 1
        progress_mon.progress(done / len(funcs))

    return res