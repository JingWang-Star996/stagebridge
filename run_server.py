"""Entrypoint supports Windows embedded Python's isolated sys.path."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
if (root / ".deps").exists():
    sys.path.insert(0, str(root / ".deps"))
from te_server.server import main
main()
