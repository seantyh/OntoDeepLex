from tqdm.autonotebook import tqdm
import re
import pickle
from joblib import Memory
from functools import partial
from itertools import chain, islice, cycle
from collections import Counter
from typing import List, Iterator, Dict, Tuple
from shelve import DbfilenameShelf
import numpy as np
import pandas as pd
import torch
import logging
from .affix_ckip import CkipAffixoids, Affixoid
from ..utils import get_data_dir, ensure_dir
from ..senses.corpus_streamer import CorpusStreamer
from ..deep.tensor_utils import BertService
from . import ctm_analysis

Word = str; Freq = int

class ByCharAnalyzer:
    def __init__(self):
        affix_dir = get_data_dir() / "affix/"
        asbc_dir = get_data_dir() / "asbc"

        logger = logging.getLogger("ByCharAnalyzer")
        self.logger = logger

        logger.info("loading charlocs")
        data_path = get_data_dir() / "affix/bychar_proc_data.pkl"
        if data_path.exists():
            with data_path.open("rb") as fin:
                data = pickle.load(fin)
        self.charlocs = data["charloc"]
        self.output_path = affix_dir / "bychar_affixoid_table.csv"

        logger.info("loading asbc5 words")
        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            self.words = pickle.load(fin)

        logger.info("loading asbc5 words with POS")
        with (asbc_dir/"asbc5_words_pos.pkl").open("rb") as fin:
            words_pos = pickle.load(fin)
            pos_set = set(p for w, p in words_pos.keys())
            self.pos_table = {p: i for i, p in enumerate(sorted(list(pos_set)))}
        self.words_pos = self.reindex_words_pos(words_pos)

        logger.info("Loading CTM")
        self.ctm = ctm_analysis.get_bychar_ctm_models()

    def reindex_words_pos(self, words_pos):        
        pos_map = {}
        for (word, pos), freq in words_pos.items():
            pos_map.setdefault(word, []).append((pos, freq))
        return pos_map

    def analyze(self):        
        
        logger = logging.getLogger("ByCharAnalyzer")
        results = []        

        for charloc, charloc_data in tqdm(self.charlocs.items()):
            try:                              
                indices = self.analyze_one(charloc, charloc_data) 
                results.append(indices)
                
            except Exception as ex: 
                import traceback
                logger.error(ex)
                logger.error(traceback.format_exc())
                break
        
        frame = pd.DataFrame.from_records(results)        
        frame.to_csv(self.output_path)

        return frame

    def analyze_one(self, charloc: str, ex_words: List[Tuple[Word, Freq]]):
        func_list = (
            self.compute_position, 
            self.compute_productivity_morph,
            self.compute_productivity_pos, 
            self.compute_meaning
            )
        
        charloc_type = "end" if charloc.index("_") == 0 else "start"
        charloc_char = charloc.replace("_", "")

        indices = {
            "affixoid": charloc,
            "affix_type": charloc_type,
            "form": charloc_char}

        for compute_func in func_list:
            compute_func(charloc, ex_words, indices)
        
        return indices

    def compute_position(self, charloc: str, 
            ex_words: List[Tuple[Word, Freq]], indices: Dict[any, any]):        
        charloc_pos = 0 if charloc.index("_") == 0 else 1
        indices.update({
            'nword': len(ex_words),
            'isstart': len(ex_words) if charloc_pos==0 else 0,
            'isend': len(ex_words) if charloc_pos==1 else 0
            })

    def compute_productivity_morph(self, 
            charloc: str, 
            ex_words: List[Tuple[Word, Freq]], 
            indices: Dict[any, any]):        
        
        if ex_words:
            wfreq = np.array([x[1] for x in ex_words])    
            log_ex_wfreq = np.log(sum(wfreq)+1)
            ex_wfreq = sum(wfreq)
            morph_index = np.log(sum(1/f for f in wfreq if f))
        else:
            ex_wfreq = 0
            morph_index = np.nan
        indices.update({
            "ex_wfreq": ex_wfreq,
            "log_ex_wfreq": log_ex_wfreq,            
            "prod_morph": morph_index
            })

    def compute_productivity_pos(self, charloc: str, 
            ex_words: List[Tuple[Word, Freq]], 
            indices: Dict[any, any]):
        ex_words = [x[0] for x in ex_words]        
        pos_counter = Counter()
        for w in ex_words:
            pos_counter.update(self.words_pos.get(w, []))

        pos_distr = list(pos_counter.values())
        pos_prob = (pos_distr / np.sum(pos_distr))
        entropy = -np.sum(np.log(pos_prob) * pos_prob)

        indices.update({"pos_entropy": entropy})        

    def compute_meaning(self, charloc: str, 
            ex_words: List[Tuple[Word, Freq]], 
            indices: Dict[any, any]):
        if self.ctm.vocab.encode(charloc):
            charloc_entropy = self.ctm.get_charloc_entropy(charloc)
        else:
            charloc_entropy = np.nan
        indices.update({"ctm_entropy": charloc_entropy})

