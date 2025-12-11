import sys
from pathlib import Path

# Ensure local package is imported before any installed version
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
