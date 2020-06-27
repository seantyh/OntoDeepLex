from argparse import ArgumentParser
import logging
from pathlib import Path
import pickle
import sys

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

ctm_dir = base_path / "../src/ctm"
assert ctm_dir.exists()
sys.path.append(str(ctm_dir))

#pylint: disable=import-error
from contextualized_topic_models.models.ctm import CTM
import mesh
from mesh.affix import AffixoidCtmDataset, ByCharCtmDataset

def main(args):
    model_type = args.type
    if model_type == "affixoid":
        vocab_path = mesh.get_data_dir() / "affix/affixoid_ctm_vocab.pkl"
        ctm_dataset = AffixoidCtmDataset()
        model_dir = mesh.get_data_dir() / "affix/affixoid"        
    elif model_type == "bychar":
        vocab_path = mesh.get_data_dir() / "affix/bychar_ctm_vocab.pkl"
        ctm_dataset = ByCharCtmDataset()
        model_dir = mesh.get_data_dir() / "affix/bychar"
    else:
        print("[ERROR] unsupported type")
        return 

    with vocab_path.open("rb") as fin:
        vocab = pickle.load(fin)

    ctm = CTM(input_size=len(vocab), bert_input_size=768, 
        inference_type="contextual", n_components=50,
        num_epochs=30)
    
    mesh.ensure_dir(model_dir)
    ctm.fit(ctm_dataset, save_dir=str(model_dir)) # run the model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", choices=["affixoid", "bychar"])
    args = parser.parse_args()
    main(args)
