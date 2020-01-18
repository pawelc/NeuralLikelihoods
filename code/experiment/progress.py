from IPython.core.display import display
from ipywidgets import FloatProgress

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # , end='\r'
    # Print New Line on Complete
    # if iteration == total:
    #     print()


class ProgressMonitor:
    def __init__(self, maximum, name):
        self._progress = None
        self._name = name
        self._max = maximum
        if is_interactive():
            self._progress = FloatProgress(min=0, max=maximum)
            display((self._progress,))
        else:
            print_progress_bar(0, maximum, prefix='Progress experiment {name}'.format(name=self._name),
                               suffix='Complete', length=50)

    def progress(self, done):
        if self._progress is not None:
            self._progress.value = done
        else:
            print_progress_bar(done, self._max, prefix='Progress experiment {name}:'.format(name=self._name),
                               suffix='Complete', length=50)

class NoOpProgressMonitor:
    def __init__(self):
        pass

    def progress(self, done):
        pass