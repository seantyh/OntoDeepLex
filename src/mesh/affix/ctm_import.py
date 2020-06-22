from pathlib import Path
import sys

base_dir = Path(__file__).parent
ctm_dir = base_dir / "../../ctm"
assert ctm_dir.exists()
if str(ctm_dir) not in sys.path:
    sys.path.append(str(ctm_dir))