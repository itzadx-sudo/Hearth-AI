import os
import platform
import shutil
import sys

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
elif getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(BASE_DIR, filename)

_app = "HearthAI"

if platform.system() == "Windows":
    DATA_DIR = os.path.join(os.getenv('APPDATA', os.path.expanduser("~")), _app)
elif platform.system() == "Darwin":
    DATA_DIR = os.path.join(os.path.expanduser("~"), "Library", "Application Support", _app)
else:
    DATA_DIR = os.path.join(os.path.expanduser("~"), f".{_app.lower()}")

os.makedirs(DATA_DIR, exist_ok=True)

def _data_path(filename):
    target = os.path.join(DATA_DIR, filename)
    if not os.path.exists(target):
        bundled = _path(filename)
        if os.path.exists(bundled):
            shutil.copy2(bundled, target)
    return target
