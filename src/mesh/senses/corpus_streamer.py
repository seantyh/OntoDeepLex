#pylint: disable=import-error
from .asbc import Asbc5Corpus
from ..utils import get_data_dir, ensure_dir
from tqdm.autonotebook import tqdm
from itertools import chain
import shelve
import pickle

class CorpusStreamer:
    def __init__(self):
        self.sentences = []
        self.char_indices = {}
        try:
            print("loading ASBC corpus streamer index")
            self.load_index()
        except FileNotFoundError:
            print("buliding ASBC corpus streamer index")
            self.build_index()        
    
    def load_index(self):
        corpus_dir = get_data_dir() / "corpus"
        with open(corpus_dir/"sentences.pkl", "rb") as fin:
            self.sentences = pickle.load(fin)
        self.char_indices = shelve.DbfilenameShelf(
                        str(corpus_dir/'char.index'), "r", 
                        writeback=False)

    def build_index(self):
        asbc = Asbc5Corpus()
        sentences = []
        
        corpus_dir = get_data_dir() / "corpus"
        ensure_dir(corpus_dir)
        char_indices = shelve.DbfilenameShelf(str(corpus_dir/'char.index'), 
                        writeback=True)

        for sent_i, sent_x in tqdm(enumerate(asbc.iter_sentences())):                                
            iter_word = map(lambda x: x[0], sent_x)                        
            for charac in chain.from_iterable(iter_word):                
                char_indices.setdefault(charac, []).append(sent_i)
            sentences.append(sent_x)
        
        char_indices.sync()
        self.char_indices = char_indices
        self.sentences = sentences
        with open(corpus_dir/"sentences.pkl", "wb") as fout:
            pickle.dump(sentences, fout)
    
    def query(self, term):
        sent_indices = [self.char_indices.get(x, []) for x in term]
        int_indices = set(chain.from_iterable(sent_indices))
        sent_iter = map(lambda idx: self.sentences[idx], int_indices)
        sent_iter = filter(lambda sent: term in (x[0] for x in sent), sent_iter)
        return sent_iter
        
