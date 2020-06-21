from argparse import ArgumentParser
import logging
from pathlib import Path
import pickle
import sys

base_path = Path(__file__).parent
mesh_src_path = (base_path / "../src").resolve().absolute()
sys.path.append(str(mesh_src_path))

ctm_dir = base_path / "../../ctm"
assert ctm_dir.exists()
sys.path.append(str(ctm_dir))

#pylint: disable=import-error
from contextualized_topic_models.models.ctm import CTM
import mesh
from mesh.affix import AffixoidCtmDataset

def main():
    vocab_path = mesh.get_data_dir() / "affix/affixoid_ctm_vocab.pkl"
    with vocab_path.open("rb") as fin:
        vocab = pickle.load(fin)

    ctm = CTM(input_size=len(vocab), bert_input_size=768, 
        inference_type="contextual", n_components=100)

    ctm_dataset = AffixoidCtmDataset()
    ctm.fit(ctm_dataset) # run the model

main()