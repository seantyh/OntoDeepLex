from argparse import ArgumentParser
import sys
from pathlib import Path
from tqdm import tqdm
from functools import partial, lru_cache

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

#pylint: disable=import-error
import mesh
from mesh.senses import SenseTagger, CorpusStreamer
import json

def write_to_json(data, suffix):
    wsd_dir = mesh.get_data_dir() / "senses/wsd"
    mesh.ensure_dir(wsd_dir)
    fname = wsd_dir/f"wsd_asbc_sentences_{suffix:05d}.json"
    print(f"Output data to {fname}")
    with open(fname, "w", encoding="UTF-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)

@lru_cache
def has_done(index):    
    wsd_dir = mesh.get_data_dir() / "senses/wsd"
    fname = wsd_dir/f"wsd_asbc_sentences_{index:05d}.json"
    return fname.exists()

def main(args):
    print("Sense tagging main routine")
    tagger = SenseTagger()
    streamer = CorpusStreamer()
    N_CHECK = 10

    buffer = []
    for sent_i, sent_x in enumerate(streamer.sentences):
        batch_index = (sent_i//N_CHECK)*N_CHECK
        if has_done(batch_index):
            print(f"{sent_i} has done, continue")
            continue

        wsd_tokens = []
        try:
            wsd_results = tagger.tag([x[0] for x in sent_x])

            for x, sense_x in zip(sent_x, wsd_results):
                if sense_x is None:
                    wsd_tokens.append((*x, "", ""))
                else:
                    wsd_tokens.append((*x, sense_x.id, sense_x.definition))
        except Exception as ex:
            print(ex)

        buffer.append([sent_i, wsd_tokens])

        if (sent_i+1) % N_CHECK == 0:
            write_to_json(buffer, batch_index)
            buffer = []
            break

if __name__ == "__main__":
    parser = ArgumentParser()

    args = parser.parse_args()
    main(args)