import sys
from pathlib import Path

base_dir = Path(__file__).parent
ctm_dir = base_dir / "../../ctm"
assert ctm_dir.exists()
if str(ctm_dir) not in sys.path:
    sys.path.append(str(ctm_dir))

#pylint: disable=import-error
from contextualized_topic_models.models.ctm import CTM

import logging
import random
import pickle
from itertools import chain, cycle, islice
from typing import List, Iterator, Iterable, Tuple, Dict
from collections import UserDict
import numpy as np
import torch
from torch.utils.data import Dataset
from .affix_ckip import CkipAffixoids, Affixoid
from ..utils import get_data_dir, ensure_dir
from .bert_service import BertService
from ..senses.corpus_streamer import CorpusStreamer

class Vocabulary(UserDict):
    UNK_ID = 0
    def __init__(self, init_data: Iterable[str]):
        super().__init__()
        UNK_ID = type(self).UNK_ID
        self.data["<UNK>"] = UNK_ID
        for x in init_data:
            self.data[x] = len(self.data)
        self.wordlist = ["UNK"] + list(init_data)        

    def update(self, new_data):
        for x in new_data:
            self.data[x] = len(self.data)
        self.wordlist += list(new_data)        

    def encode(self, word):
        if isinstance(word, str):
            return self.data.get(word, type(self).UNK_ID)
        elif isinstance(word, list):
            return [self.encode(x) for x in word]
        else:
            raise TypeError()

    def decode(self, id):
        if isinstance(id, int):
            if 0 <= id < len(self.wordlist):
                return self.wordlist[id]
            else:
                raise IndexError()
        elif isinstance(id, list):
            return [self.decode(x) for x in id]
        else:
            raise TypeError()


class AffixoidCtmProcessor:
    def __init__(self):        
        affix_dir = get_data_dir() / "affix/"        
        self.logger = logging.getLogger(__name__)

        self.logger.info("loading CkipAffixoids")
        self.affixoids = CkipAffixoids(affix_dir)
        self.logger.info("loading Asbc Corpus")
        # self.asbc = CorpusStreamer()
        self.logger.info("loading DistilBert")
        # self.bert = BertService()
        
        asbc_dir = get_data_dir() / "asbc"
        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            words = pickle.load(fin)
        self.vocab = self.build_vocab(words, self.affixoids)
    
    def build_examples(self):
        n_ex_words = sum(len(x.example_words) for x in self.affixoids)
        ex_words_iter = (zip(cycle([x]), x.example_words) for x in self.affixoids)
        ex_words_iter = chain.from_iterable(ex_words_iter)
        ex_words_iter = map(lambda x: (x[0].affixoid, x[1][1]), ex_words_iter)

        logger = self.logger
        random.seed(12345)
        examples = []        
        for idx, (affixoid, word) in enumerate(ex_words_iter):            
            self.build_word_examples(affixoid, word)
        
        return examples

    def build_word_examples(self, affixoid:str, word: str):
        try:                        
            sent_iter = self.asbc.query(word)
            n = 0
            batch_size = 20
            MAX_SIZE = 200
            for first in sent_iter:
                batch_iter = chain(first, islice(sent_iter, batch_size-1))
                self.build_single_example(affixoid, word, batch_iter)                
                n += batch_size
                if n >= MAX_SIZE:
                    break
        except Exception as ex: 
            import traceback
            self.logger.error(ex)
            self.logger.error(traceback.format_exc())

    def build_single_example(self, 
            affixoid: str, 
            target_word: str,
            sentences: Iterator[str]):
        sent_text = [''.join(x[0] for x in sent) for sent in sentences]
        bow_text = [self.build_bow(sent) for sent in sentences]
        targ_indices = [x.index(target_word) + target_word.index(affixoid)
                        for x in sent_text]
        input_data = self.bert.encode(sentences)

        #pylint: disable=not-callable
        input_tensors = {k: torch.tensor(v) for k, v in input_data.items()}
        targ_token_indices = [input_data.char_to_token(b, i)
                                for b, i in enumerate(targ_indices)]
        outputs = self.bert.transform(input_tensors)
        affix_vecs = outputs[np.arange(len(targ_token_indices)), targ_token_indices, :]
        return affix_vecs

    Word = str; POS = str
    from collections import Counter
    def build_vocab(self, words: Dict[Word, int], affixoids: CkipAffixoids):
        word_iter = [w for (w, f) in words.items() if f > 50]
        vocab = Vocabulary(word_iter)
        affix_iter = (aff.affix_form() for aff in affixoids)        
        vocab.update(list(affix_iter))
        return vocab

    def build_bow(self, sentence:List[Tuple[Word, POS]]):
        self.vocab

class AffixoidCtmDataset(Dataset):
    def __int__(self, ex_path=None):
        logger = logging.getLogger(__name__)
        if ex_path is None:
            ex_path = get_data_dir() / "affix/affixoid_ctm_examples.pkl"
        if ex_path.exists():
            fin = ex_path.open("rb")
            self.examples = pickle.load(fin)
            fin.close()
        else:
            logger.info("Building Affixoid CTM examples...")
            ex_processor = AffixoidCtmProcessor()
            self.examples = ex_processor.build()


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]