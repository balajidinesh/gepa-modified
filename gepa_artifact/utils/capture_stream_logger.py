import sys
import os
import datetime

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(hasattr(s, "isatty") and s.isatty() for s in self.streams)


class Logger:
    def __init__(self, filename, mode="a", also_print=True):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.log_file = open(filename, mode, buffering=1)
        self.err_file = open(filename.replace("run_log", "run_log_stderr"), mode, buffering=1)

        self.also_print = also_print

        # Save originals
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        # Redirect automatically
        sys.stdout = Tee(sys.stdout, self.log_file) if also_print else self.log_file
        sys.stderr = Tee(sys.stderr, self.err_file) if also_print else self.err_file

    def log(self, *args, sep=" ", end="\n"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        message = sep.join(str(a) for a in args)
        formatted = f"[{timestamp}] {message}"

        # Console (if enabled)
        if self.also_print:
            self._stdout.write(formatted + end)
            self._stdout.flush()

        # File
        self.log_file.write(formatted + end)
        self.log_file.flush()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.log_file.close()
        self.err_file.close()
