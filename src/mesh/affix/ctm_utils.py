import sys
from pathlib import Path

base_dir = Path(__file__).parent
ctm_dir = base_dir / "../../ctm"
assert ctm_dir.exists()
if str(ctm_dir) not in sys.path:
    sys.path.append(str(ctm_dir))

from contextualized_topic_models.models.ctm import CTM

import logging
import random
from itertools import chain, cycle
from torch.utils.data import Dataset
from .affix_ckip import CkipAffixoids, Affixoid
from ..utils import get_data_dir, ensure_dir
from ..senses.corpus_streamer import CorpusStreamer

class AffixoidCtmExamples:
    def __init__(self):        
        affix_dir = get_data_dir() / "affix/"        
        self.logger = logging.getLogger("CtmAffixoidDataset")

        self.logger.info("loading CkipAffixoids")
        self.affixoids = CkipAffixoids(affix_dir)
        self.logger.info("loading Asbc Corpus")
        self.asbc = CorpusStreamer()
        self.logger.info
        self.bert = 
    
    def build_examples(self):
        n_ex_words = sum(len(x.example_words) for x in self.affixoids)
        ex_words_iter = (zip(cycle([x]), x.example_words) for x in self.affixoids)
        ex_words_iter = chain.from_iterable(ex_words_iter)
        ex_words_iter = map(lambda x: (x[0].affixoid, x[1][1]), ex_words_iter)

        logger = self.logger
        random.seed(12345)
        examples = []
        N_MAX_SENTENCE = 200
        for idx, (affixoid, word) in enumerate(ex_words_iter):            
            try:                
                sent_iter = islice(self.asbc.query(word)), 200)
                sentences = list(x[0] for x in sent_iter)
                
                self.build_simple_example()
                break
            except Exception as ex: 
                import traceback
                logger.error(ex)
                logger.error(traceback.format_exc())
        
        self.examples = examples

    def build_simple_example(self, affixoid: str, sentences: List[str]):
        sent_text = [''.join(x[0] for x in sent) for sent in sentences]
        targ_indices = [x.index(target_word) + target_word.index(affixoid)
                        for x in sent_text]
        input_data = self.bert.encode(sentences)

        #pylint: disable=not-callable
        input_tensors = {k: torch.tensor(v) for k, v in input_data.items()}
        targ_token_indices = [input_data.char_to_token(b, i)
                                for b, i in enumerate(targ_indices)]
        outputs = self.bert.transform(input_tensors)
        affix_vecs = outputs[0][np.arange(len(targ_token_indices)), targ_token_indices, :]
        return affix_vecs

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return []