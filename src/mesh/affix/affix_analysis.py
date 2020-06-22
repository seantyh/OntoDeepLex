from tqdm.autonotebook import tqdm
import re
import pickle
from joblib import Memory
from functools import partial
from itertools import chain, islice, cycle
from collections import Counter
from typing import List, Iterator, Dict
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

class AffixoidAnalyzer:
    def __init__(self):
        affix_dir = get_data_dir() / "affix/"
        asbc_dir = get_data_dir() / "asbc"

        logger = logging.getLogger("AffixAnalyzer")
        self.logger = logger

        logger.info("loading CkipAffixoids")
        self.affixoids = CkipAffixoids(affix_dir)
        self.output_path = affix_dir / "affixoid_table.csv"

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
        self.ctm = ctm_analysis.get_ctm_models()

    def reindex_words_pos(self, words_pos):        
        pos_map = {}
        for (word, pos), freq in words_pos.items():
            pos_map.setdefault(word, []).append((pos, freq))
        return pos_map

    def analyze(self):        
        
        logger = logging.getLogger("AffixAnalyzer")
        results = []        

        for affixoid in tqdm(self.affixoids):
            try:                
                indices = self.analyze_one(affixoid) 
                results.append(indices)
                
            except Exception as ex: 
                import traceback
                logger.error(ex)
                logger.error(traceback.format_exc())
        
        frame = pd.DataFrame.from_records(results)        
        frame.to_csv(self.output_path)

        return frame

    def analyze_one(self, affixoid: Affixoid):
        func_list = (
            self.compute_position, 
            self.compute_productivity_morph,
            self.compute_productivity_pos, 
            self.compute_meaning
            )

        indices = {
            "affixoid": affixoid.affix_form(),
            "affix_type": affixoid.affixoid_type,
            "form": affixoid.affixoid}

        for compute_func in func_list:
            compute_func(affixoid, indices)
        
        return indices

    def compute_position(self, affixoid:Affixoid, indices: Dict[any, any]):
        ex_words = [x[1] for x in affixoid.example_words]
        indices.update({
            'nword': len(ex_words),
            'isstart': len(ex_words) if affixoid.position==0 else 0,
            'isend': len(ex_words) if affixoid.position==1 else 0
            })

    def compute_productivity_morph(self, affixoid: Affixoid, indices: Dict[any, any]):
        ex_words = [x[1] for x in affixoid.example_words]
        
        if ex_words:
            wfreq = np.array([self.words.get(w, 0) for w in ex_words])    
            ex_wfreq = np.log(sum(wfreq)+1)
            morph_index = np.log(sum(1/f for f in wfreq if f))
        else:
            ex_wfreq = 0
            morph_index = np.nan
        indices.update({
            "ex_wfreq": ex_wfreq,
            "prod_morph": morph_index
            })

    def compute_productivity_pos(self, affixoid: Affixoid, indices: Dict[any, any]):
        ex_words = [x[1] for x in affixoid.example_words]        
        pos_counter = Counter()
        for w in ex_words:
            pos_counter.update(self.words_pos.get(w, []))

        pos_distr = list(pos_counter.values())
        pos_prob = (pos_distr / np.sum(pos_distr))
        entropy = -np.sum(np.log(pos_prob) * pos_prob)

        indices.update({"pos_entropy": entropy})        

    def compute_meaning(self, affixoid: Affixoid, indices: Dict[any, any]):
        aff_entropy = self.ctm.get_affixoid_entropy(affixoid)
        indices.update({"ctm_entropy": aff_entropy})

