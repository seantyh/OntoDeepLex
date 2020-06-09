
from pathlib import Path
import sys

basepath = Path(__file__).parent / "../../../../pyASBC/src"
sys.path.append(str(basepath.resolve().absolute()))

#pylint: disable=import-error
from pyASBC import Asbc5Corpus

    