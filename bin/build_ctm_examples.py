from argparse import ArgumentParser
import logging
from pathlib import Path
import sys

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

from mesh.affix import AffixoidCtmProcessor
logger = logging.getLogger()
logging.basicConfig(level="INFO", format="[%(levelname)s] (%(asctime)s) %(module)s: %(message)s")

if __name__ == "__main__":
    parser = ArgumentParser()    
    parser.add_argument("--n", default=0, type=int,
            help="number to build, 0 to build all examples")
    args = parser.parse_args()
    
    proc = AffixoidCtmProcessor()
    proc.build_examples(n_to_build=args.n)    
    proc.export_examples()
    proc.close()

