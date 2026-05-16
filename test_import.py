import faulthandler
import signal
import sys
import os

faulthandler.enable()

def handler(signum, frame):
    os.write(2, b"Timeout! Dumping tracebacks...\n")
    faulthandler.dump_traceback(sys.stderr)
    os._exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(3)

import main
