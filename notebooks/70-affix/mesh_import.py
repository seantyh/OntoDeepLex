import sys
from pathlib import Path

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

import mesh