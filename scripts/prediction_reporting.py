"""Standalone reporting entry point; the shared implementation is in src."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cacao_coffee_panels.reporting import *
if __name__ == '__main__':
    raise SystemExit(main())
