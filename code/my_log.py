import logging.handlers
import os
import sys
init_logging_done = False

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def init_logging(file):
    global init_logging_done
    if not init_logging_done:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        handler = logging.handlers.WatchedFileHandler(
            os.environ.get("LOGFILE", file))
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        root.addHandler(handler)

        init_logging_done = True
        logging.info("Logging initialized")

        # log_stdout = logging.getLogger('stdout')
        # log_stderr = logging.getLogger('stderr')
        # sys.stdout = LoggerWriter(log_stdout.info)
        # sys.stderr = LoggerWriter(log_stderr.warning)

