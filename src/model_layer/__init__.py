import sys
import os

# Add /kaggle/working to Python path
if '/kaggle/working' not in sys.path:
    sys.path.insert(0, '/kaggle/working')

# Add parent directories
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)