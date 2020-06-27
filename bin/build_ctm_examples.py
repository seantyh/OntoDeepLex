from argparse import ArgumentParser
import logging
from pathlib import Path
import sys

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

#pylint: disable=import-error
from mesh.affix import AffixoidCtmProcessor
from mesh.affix import ByCharCtmProcessor

logger = logging.getLogger()
logging.basicConfig(level="INFO", format="[%(levelname)s] (%(asctime)s) %(module)s: %(message)s")

if __name__ == "__main__":
    parser = ArgumentParser()    
    parser.add_argument("--n", default=0, type=int,
            help="number to build, 0 to build all examples")
    parser.add_argument("--type", choices=["affixoid", "bychar"])
    args = parser.parse_args()
    
    if args.type == "affixoid":
        proc = AffixoidCtmProcessor()
        proc.build_examples(n_to_build=args.n)        
        proc.close()
    elif args.type == "bychar":
        proc = ByCharCtmProcessor()
        proc.build_examples(n_to_build=args.n)
    else:
        parser.print_help()

