from transformers import PreTrainedTokenizer
from ..utils import get_data_dir
import pickle
import re
from typing import Dict
from zhon import hanzi


class VocabZhTw:
    def __init__(self):
        self.vocab: Dict[str, int] = self.load_vocab()
        self.itos: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def load_vocab(self):
        asbc_dir = get_data_dir() / "asbc"
        with (asbc_dir/"asbc5_characters.pkl").open("rb") as fin:
            as_chfreq = pickle.load(fin).most_common()
        cjk_pat = re.compile(f"[{hanzi.characters}]")
        as_chfreq = filter(lambda x: cjk_pat.match(x[0]), as_chfreq)
        vocab = {k[0]: i for i, k in enumerate(as_chfreq)}
        return vocab
    
    def encode(self, text):
        return [self.vocab.get(x, len(self.vocab)) for x in text]

    def decode(self, ids):
        return [self.itos.get(i, "<UNK>") for i in ids]

    def get_mapping(self, tgt_vocab):
        """
        return mappings from target vocabulary to this vocabulary

        Return
        ------
        target index: the index in target vocabulary
        this index: the corresponding index in this vocabulary
        """
        mapping = ((tgt_vocab.get(wd), wd_idx) for wd, wd_idx in self.vocab.items()
                    if wd in tgt_vocab)
        mapping = sorted(mapping, key=lambda x: x[1])
        tgt_idx, this_idx = zip(*mapping)
        return tgt_idx, this_idx

    def __contains__(self, key):
        return key in self.vocab

