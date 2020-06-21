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
from functools import partial
from typing import List, Iterator, Iterable, Tuple, Dict
from collections import UserDict, Counter
from shelve import DbfilenameShelf
from tqdm.autonotebook import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from .affix_ckip import CkipAffixoids, Affixoid
from ..utils import get_data_dir, ensure_dir
from .bert_service import BertService
from ..senses.corpus_streamer import CorpusStreamer

Word = str; POS = str
class Vocabulary:
    UNK_ID = 0
    def __init__(self, init_data: Iterable[str]):        
        UNK_ID = type(self).UNK_ID
        self.data = {}
        for x in init_data:
            self.data[x] = len(self.data)
        self.wordlist = list(init_data)

    def __len__(self):
        return len(self.wordlist)

    def update(self, new_data):
        for x in new_data:
            self.data[x] = len(self.data)
        self.wordlist += list(new_data)        

    def encode(self, word):        
        if isinstance(word, str):
            return self.data.get(word, None)
        elif isinstance(word, list):
            return [self.encode(x) for x in word if x in self.data]
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
        self.asbc = CorpusStreamer()
        self.logger.info("loading DistilBert")
        self.bert = BertService()

        ensure_dir(affix_dir/"ctm_examples")
        self.example_store = DbfilenameShelf(str(affix_dir/"ctm_examples/ctm.examples"), writeback=True)
        
        asbc_dir = get_data_dir() / "asbc"
        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            words = pickle.load(fin)
        self.vocab = self.build_vocab(words, self.affixoids)

    
    def close(self):
        self.example_store.close()
    
    def save_vocabulary(self):
        vocab_path = get_data_dir() / "affix/affixoid_ctm_vocab.pkl"
        with vocab_path.open("wb") as fout:
            pickle.dump(self.vocab, vocab_path)

    def build_examples(self, n_to_build=0):
        n_ex_words = sum(len(x.example_words) for x in self.affixoids)
        ex_words_iter = (zip(cycle([x]), x.example_words) for x in self.affixoids)
        ex_words_iter = chain.from_iterable(ex_words_iter)
        ex_words_iter = map(lambda x: (x[0], x[1][1]), ex_words_iter)

        logger = self.logger
        random.seed(12345)
        
        n_done = 0
        for idx, (affixoid, word) in enumerate(ex_words_iter):    
            aff_word = affixoid.affixoid
            self.logger.info("[%d/%d] building example %s/%s", 
                idx, n_ex_words, aff_word, word)        
            store_key = repr((aff_word, word))
            if store_key in self.example_store:
                continue
            try:
                examples = self.build_affixoid_examples(affixoid, word)
                self.example_store[store_key] = examples                
                self.example_store.sync()
                n_done += 1

            except Exception as ex:
                import traceback
                logger.error(ex)
                logger.error(traceback.format_exc())            

            if n_to_build and n_done >= n_to_build:
                logger.info("Build finished for %d examples to build", n_done)
                break
        

    def export_examples(self):        
        ex_path = get_data_dir() / "affix/affixoid_ctm_examples.pkl"
        self.logger.info("exporting examples to %s", ex_path)

        fout = ex_path.open("wb")
        examples = list(chain.from_iterable(self.example_store.values()))
        pickle.dump(examples, fout)
        fout.close()

        self.logger.info("Export done")

    def build_affixoid_examples(self, affixoid:Affixoid, word: str):
        try:                        
            sent_iter = self.asbc.query(word)            
            n = 0
            batch_size = 20
            MAX_SIZE = 20

            # by-batch loop

            bows_list = []; bert_vlist = []
            for first in sent_iter:                                
                batch_iter = chain([first], islice(sent_iter, batch_size-1))                               
                bows, bert_vecs = self.build_batch_example(affixoid, word, batch_iter)                
                bows_list.append(bows)
                bert_vlist.append(bert_vecs)
                n += batch_size
                if n >= MAX_SIZE:
                    break
                        
            out_examples = list(zip(
                    chain.from_iterable(bows_list), 
                    chain.from_iterable(bert_vlist)))

            return out_examples
        except Exception as ex: 
            import traceback
            self.logger.error(ex)
            self.logger.error(traceback.format_exc())

    def build_batch_example(self, 
            affixoid: Affixoid, 
            target_word: str,
            sentences: Iterator[Tuple[POS, Word]]):
        sentences = list(sentences)        
        sent_text = [''.join(x[0] for x in sent) for sent in sentences]        
        build_bow_func = partial(self.build_bow, affixoid=affixoid)
           
        bow_list = [build_bow_func(sent) for sent in sentences]        
        targ_indices = [x.index(target_word) + target_word.index(affixoid.affixoid)
                        for x in sent_text]
        input_tensors = self.bert.encode(sent_text)

        #pylint: disable=not-callable        
        targ_token_indices = [input_tensors.char_to_token(b, i)
                                for b, i in enumerate(targ_indices)]
        outputs = self.bert.transform(input_tensors)        
        affix_vecs = outputs[np.arange(len(targ_token_indices)), targ_token_indices, :]
        affix_vecs = list(affix_vecs.numpy())
        return bow_list, affix_vecs

    def build_vocab(self, words: Dict[Word, int], affixoids: CkipAffixoids):
        word_iter = [w for (w, f) in words.items() if f > 100]
        vocab = Vocabulary(word_iter)
        affix_iter = (aff.affix_form() for aff in affixoids)        
        vocab.update(list(affix_iter))
        return vocab

    def build_bow(self, sentence:List[Tuple[Word, POS]], affixoid: Affixoid):
        vocab = self.vocab           
        sids = vocab.encode([x[0] for x in sentence])        
        sids.append(vocab.encode(affixoid.affix_form()))

        sid_bow = np.bincount(sids, minlength=len(vocab))
        return sid_bow


        

class AffixoidCtmDataset(Dataset):
    def __init__(self, ex_path=None):
        self.logger = logging.getLogger(__name__)
        affix_dir = get_data_dir() / "affix"
        self.example_store = DbfilenameShelf(
            str(affix_dir/"ctm_examples/ctm.examples"), writeback=False)
        self.store_index = self.index_store()

    def index_store(self):
        self.logger.info("indexing store")
        store_index = []
        counter = 0
        for store_key, data in tqdm(self.example_store.items()):            
            store_index.extend([(store_key, i) for i, _ in enumerate(data)])
        return store_index

    def __len__(self):
        return len(self.store_index)

    def __getitem__(self, idx):
        idx = self.store_index[idx]
        return self.example_store[idx[0]][idx[1]]
