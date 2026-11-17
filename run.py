"""Run from a source checkout without installing a new environment."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"1")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cacao_coffee_panels.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
