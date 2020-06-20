from argparse import ArgumentParser
import logging
from pathlib import Path
import sys

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

from mesh.affix import AffixoidAnalyzer
logger = logging.getLogger()
logging.basicConfig(level="INFO", format="[%(levelname)s] %(module)s: %(message)s")

if __name__ == "__main__":
    parser = ArgumentParser()    
    args = parser.parse_args()
    
    analyzer = AffixoidAnalyzer(analysis=True)
    analyzer.analyze()

