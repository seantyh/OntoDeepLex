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

class WordAnalyzer:   
    """Deprecated. 
    originally used with AffixoidAnalyzer. Keep for reference
    """
    def __init__(self):
        asbc_dir = get_data_dir() / "asbc"
        self.asbc = CorpusStreamer()
        self.bert = BertService()
        logger = logging.getLogger("WordAnalyzer")

        logger.info("loading asbc5 words")
        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            self.words = pickle.load(fin)

        logger.info("loading asbc5 words with POS")
        with (asbc_dir/"asbc5_words_pos.pkl").open("rb") as fin:
            self.words_pos = pickle.load(fin)

        self.words_pos = self.reindex_words_pos()

    def reindex_words_pos(self):
        words_pos = self.words_pos
        pos_map = {}
        for (word, pos), freq in words_pos.items():
            pos_map.setdefault(word, []).append((pos, freq))
        return pos_map

    def normalize_indices(self, options):
        if options is None:
            return None

        norm_opts = set()
        for opt in options:
            opt = opt.lower()
            if re.search("pos\b", opt):
                norm_opts.add("prod_pos")
            elif re.search("morph", opt):
                norm_opts.add("prod_morph")
            elif re.search("position", opt):
                norm_opts.add("position")
            elif re.search("meaning|semantic", opt):
                norm_opts.add("meaning")

        return norm_opts

    def analyze(self, affixoid: str, word: str, indices=None):
        indices = self.normalize_indices(indices)
        if not indices or "position" in indices:
            position_data = self.compute_position(affixoid, word)

        if not indices or "prod_morph" in indices:
            morph_data = self.compute_productivity_morph(affixoid, word)

        if not indices or "prod_pos" in indices:
            pos_data = self.compute_productivity_pos(affixoid, word)

        if not indices or "meaning" in indices:
            meaning_data = self.compute_meaning(affixoid, word)

        return {
            "position": position_data,
            "prod_morph": morph_data,
            "prod_pos": pos_data,
            "meaning": meaning_data
        }

    def compute_position(self, affixoid:str, word: str):
        if word.startswith(affixoid):
            return 0
        elif word.endswith(affixoid):
            return 1
        else:
            return -1

    def compute_productivity_morph(self, affixoid:str, word: str):
        return self.words.get(word, 0)

    def compute_productivity_pos(self, affixoid:str, word: str):
        return self.words_pos.get(word, [])

    def compute_meaning(self, affixoid:str, word: str):
        sent_iter = self.asbc.query(word)
        candidates = self.compute_mlm_candidates(affixoid, word, sent_iter)
        # self.asbc.query(affixoid)
        return candidates

    def compute_mlm_candidates_batch(self,
            sentence_iter: Iterator[str],
            affixoid: str,
            target_word: str):
        sentences = [''.join(x[0] for x in sent) for sent in sentence_iter]
        targ_indices = [x.index(target_word) + target_word.index(affixoid)
                        for x in sentences]
        input_data = self.bert.encode(sentences)

        #pylint: disable=not-callable
        input_tensors = {k: torch.tensor(v) for k, v in input_data.items()}
        targ_token_indices = [input_data.char_to_token(b, i)
                                for b, i in enumerate(targ_indices)]
        outputs = self.bert.transform(input_tensors, k=10)
        if outputs[1].ndim == 2:
            predicted = outputs[1][targ_token_indices, :]
        else:
            predicted = outputs[1][np.arange(len(targ_token_indices)), targ_token_indices, :]
        candidates = [self.bert.decode(x).split() for x in predicted]
        return candidates

    def compute_mlm_candidates(self, affixoid: str,
            target_word: str, sentences: Iterator[str]):
        logger = logging.getLogger("WordAnalyzer")
        def batch(iterable, size=20, max_size=200):
            iterator = iter(iterable)
            n = 0
            for first in iterator:
                yield chain([first], islice(iterator, size - 1))
                n += size
                logger.info("working on mlm candidates: %d", n)
                if n >= max_size:
                    break

        compute_func = partial(
            self.compute_mlm_candidates_batch,
            affixoid=affixoid, target_word=target_word)
        candidates = chain.from_iterable(compute_func(x) for x in batch(sentences))
        return list(candidates)
