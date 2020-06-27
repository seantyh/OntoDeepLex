import logging
import random
import pickle
from itertools import chain, cycle, islice
from functools import partial
from typing import List, Iterator, Iterable, Tuple, Dict
from collections import UserDict, Counter, defaultdict
from shelve import DbfilenameShelf
from tqdm.autonotebook import tqdm

import numpy as np
from .bert_service import BertService
from .ctm_utils import Vocabulary
from ..utils import get_data_dir, ensure_dir
from ..senses.corpus_streamer import CorpusStreamer
from .affix_ckip import Affixoid, CkipAffixoids

Word = str; POS = str; Freq = int
Sentence = List[Tuple[Word, POS]]
LocChar = str # "字_" or "_上來"
LocCharWords = Dict[LocChar, List[Tuple[Word, Freq]]]
LocCharSentences = Dict[LocChar, List[Tuple[Word, Sentence]]]
class ByCharCtmProcessor:
    def __init__(self, debug=False, no_cache=False):  
        self.logger = logging.getLogger(__name__)

        self.is_debug = debug
        if self.is_debug:
            self.logger.warning("Debug mode is ON")
        else:
            self.logger.info("Debug mode is OFF")

        affix_dir = get_data_dir() / "affix/"
        asbc_dir = get_data_dir() / "asbc/"
        
        self.logger.info("Loading affixoids")
        self.affixoids = CkipAffixoids()
        self.logger.info("loading Asbc Corpus")
        self.asbc = CorpusStreamer()
        self.logger.info("loading DistilBert")
        self.bert = BertService()

        ensure_dir(affix_dir/"bychar_examples")
        self.example_store = DbfilenameShelf(str(affix_dir/"bychar_examples/bychar.examples"))

        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            words = pickle.load(fin)
        self.words = words
        
        self.charlocs = None
        self.vocab = None
        self.charloc_sentences = None        

        from_cache = False
        if not no_cache:
            from_cache = self.load_artefacts()
        else:
            self.logger.warning("no_cache is TRUE, all cache is ignored")
            self.example_store.clear()

        if not self.charlocs:
            self.logger.info("build charloc")
            self.charlocs = self.build_charlocs(words)

        if not self.vocab:
            self.logger.info("Building vocabulary")
            self.vocab = self.build_vocab(words, self.charlocs)
        
        if not self.charloc_sentences:
            self.logger.info("Building charloc_sentences")
            self.charloc_sentences = self.build_charloc_sentences(self.charlocs)                
        
        if not from_cache:
            self.save_artefacts()

    def close(self):
        self.example_store.close()

    def save_artefacts(self):
        vocab_path = get_data_dir() / "affix/bychar_ctm_vocab.pkl"
        data_path = get_data_dir() / "affix/bychar_proc_data.pkl"

        with vocab_path.open("wb") as fout:
            pickle.dump(self.vocab, fout)

        with data_path.open("wb") as fout:
            pickle.dump({
                "charloc": self.charlocs,
                "charloc_sentences": self.charloc_sentences
                }, fout)
    
    def load_artefacts(self):
        vocab_path = get_data_dir() / "affix/bychar_ctm_vocab.pkl"
        data_path = get_data_dir() / "affix/bychar_proc_data.pkl"
        
        load_from_cache = True
        if vocab_path.exists():
            with vocab_path.open("rb") as fin:
                self.vocab = pickle.load(fin)
        else:
            load_from_cache = False
        
        if data_path.exists():
            with data_path.open("rb") as fin:
                data = pickle.load(fin)
            self.charlocs = data["charloc"]
            self.charloc_sentences = data["charloc_sentences"]
        else:
            load_from_cache = False

        return load_from_cache
        
    def build_charlocs(self, as_wfreq) -> LocCharWords:
        aff_chars = set(x.affixoid for x in self.affixoids)
        charpos = defaultdict(list)
        for w, f in tqdm(as_wfreq.items(), desc="building charlocs"):
            if not w:
                self.logger.info("empty word in AS wordlist")
                continue
            if len(w) == 1:
                continue
            cstart = w[0]
            cend = w[-1]
            cstart2 = w[:2]
            cend2 = w[-2:]
            w_tuple = (w,f)
            if cstart in aff_chars:
                charpos[cstart + "_"].append(w_tuple)
            if cstart2 in aff_chars:
                charpos[cstart2 + "_"].append(w_tuple)
            if cend in aff_chars:
                charpos["_" + cend].append(w_tuple)
            if cend2 in aff_chars:
                charpos["_" + cend2].append(w_tuple)

            if self.is_debug and len(charpos) > 10:
                break
        return charpos

    def build_charloc_sentences(self, charloc: LocCharWords):
        char_sentences = {}
        for char, data in tqdm(charloc.items(), desc="Sample sentences"):
            char_sentences[char] = self.sample_sentences(data)
        return char_sentences

    def sample_sentences(self,
            word_list: List[Tuple[Word, Freq]])\
            -> LocCharSentences:
        
        N_SENTENCE_PER_CHARLOC = 50
        random.seed(12345)
        np.random.seed(12345)
        freq = [x[1] for x in word_list]
        pvec = freq/np.sum(freq)
        sample_vec = np.random.multinomial(N_SENTENCE_PER_CHARLOC, pvec)
        sentences = []        
        for idx, n_sent in enumerate(sample_vec):
            if n_sent == 0:
                continue
            word = word_list[idx][0]
            sent_pool = list(islice(self.asbc.query(word), n_sent*10))
            random.shuffle(sent_pool)
            sents = sent_pool[:n_sent]
            sentences.extend(list(zip(cycle([word]), sents)))
        return sentences

    def build_examples(self, n_to_build=0):
        logger = self.logger
        random.seed(12345)

        n_done = 0
        sentence_item = self.charloc_sentences.items()
        n_item = sum(len(x) for x in self.charloc_sentences.values())
        
        for idx, (charloc, sentence_data) in enumerate(sentence_item):
            self.logger.info("[%d/%d] building example %s",
                idx, n_item, charloc)
            store_key = charloc
            if store_key in self.example_store:
                continue
            try:
                examples = self.build_sentence_examples(charloc, sentence_data)
                self.example_store[store_key] = examples                
                n_done += 1

            except Exception as ex:
                import traceback
                logger.error(ex)
                logger.error(traceback.format_exc())

            if n_to_build and n_done >= n_to_build:
                logger.info("Build finished for %d examples to build", n_done)
                break

    def build_sentence_examples(self, charloc: LocChar, 
            sentence_data: List[Tuple[Word, Sentence]]):
        try:            
            n = 0
            batch_size = 20
            MAX_SIZE = 20
            item_iter = iter(sentence_data)

            # by-batch loop            
            bows_list = []; bert_vlist = []
            for first in item_iter:
                batch_iter = chain([first], islice(item_iter, batch_size-1))
                bows, bert_vecs = self.build_batch_example(charloc, batch_iter)
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
            charloc: LocChar,            
            item_iter: Iterator[Tuple[Word, Sentence]]):
        items = list(item_iter)
        sentences = [x[1] for x in items]
        target_words = [x[0] for x in items]
        sent_text = [''.join(x[0] for x in sent) for sent in sentences]
        build_bow_func = partial(self.build_bow, charloc=charloc)

        bow_list = [build_bow_func(sent) for sent in sentences]
        charonly = charloc.replace("_", "")
        targ_indices = [sent.index(w) + w.index(charonly)
                        for w, sent in zip(target_words, sent_text)]                
        input_tensors = self.bert.encode(sent_text)

        #pylint: disable=not-callable
        targ_token_indices = [input_tensors.char_to_token(b, i)
                                for b, i in enumerate(targ_indices)]
        outputs = self.bert.transform(input_tensors)
        affix_vecs = outputs[np.arange(len(targ_token_indices)), targ_token_indices, :]
        affix_vecs = list(affix_vecs.numpy())
        return bow_list, affix_vecs

    def build_vocab(self, words: Dict[Word, int], charlocs: LocCharWords):
        word_iter = [w for (w, f) in words.items() if f > 50]
        vocab = Vocabulary(word_iter)
        charloc_iter = (charloc for charloc in charlocs.keys())
        vocab.update(list(charloc_iter))
        return vocab

    def build_bow(self, sentence:List[Tuple[Word, POS]], charloc: LocChar):
        vocab = self.vocab
        sids = vocab.encode([x[0] for x in sentence])
        sids.append(vocab.encode(charloc))              
        sid_bow = np.bincount(sids, minlength=len(vocab))
        return sid_bow