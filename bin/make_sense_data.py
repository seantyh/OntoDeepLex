from argparse import ArgumentParser
import sys
from pathlib import Path
from tqdm import tqdm
from functools import partial

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

#pylint: disable=import-error
from mesh.senses import SenseTagger, CorpusStreamer

def main(args):
    if args.characters:
        charac = args.characters
    
    tagger = SenseTagger
    streamer = CorpusStreamer()

    tag_func = partial(tagger.tag, streamer=streamer)
    resiter = map(lambda x: (x, tag_func(x)), charac)
    results = list(tqdm(resiter))
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    
    args = parser.parse_args()
    main(args)