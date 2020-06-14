from .affix_ckip import CkipAffixoids, Affixoid
from tqdm.autonotebook import tqdm
from ..utils import get_data_dir, ensure_dir
import re
import pickle
from joblib import Memory
from functools import partial

class AffixoidAnalyzer:
    def __init__(self, affixoids: CkipAffixoids):
        self.affixoids = affixoids
        asbc_dir = get_data_dir() / "asbc"        

        print("loading asbc5 words")
        with (asbc_dir/"asbc5_words.pkl").open("rb") as fin:
            self.words = pickle.load(fin)

        print("loading asbc5 words with POS")
        with (asbc_dir/"asbc5_words_pos.pkl").open("rb") as fin:
            self.words_pos = pickle.load(fin)
    
    def init_cache_function(self):
        affix_dir = get_data_dir() / "affix"
        cache_dir = affix_dir/"analyzer_cache"
        ensure_dir(cache_dir)
        mem = Memory(cache_dir)

        self.cache_position_func = mem.cache(self.compute_position)
        self.cache_prod_morph_func = mem.cache(self.compute_productivity_morph)
        self.cache_prod_pos_func = mem.cache(self.compute_productivity_pos)
        self.cache_meaning_func = mem.cache(self.compute_meaning)
    
    def analyze(self, indices=None):
        analyze_func = partial(self.analyze_one, indices=indices)
        analy_iter = map(analyze_func, self.affixoids)
        return list(tqdm(analy_iter, total=len(self.affixoids)))
    
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

    def analyze_one(self, affixoid: Affixoid, indices=None):
        indices = self.normalize_indices(indices)
        if not indices or "position" in indices:
            pos_vec = self.compute_position(affixoid)

        if not indices or "prod_morph" in indices:
            morph_vec = self.compute_productivity_morph(affixoid)

        if not indices or "prod_pos" in indices:
            morph_vec = self.compute_productivity_pos(affixoid)
        
        if not indices or "meaning" in indices:
            morph_vec = self.compute_meaning(affixoid)
            
    def compute_position(self, affixoid: Affixoid):
        if affixoid.position == 0:
            return [len(affixoid.example_words), 0, -1]
        else:
            return [0, len(affixoid.example_words), -1]        

    def compute_productivity_morph(self, affixoid: Affixoid):
        return []
    
    def compute_productivity_pos(self, affixoid: Affixoid):
        return []

    def compute_meaning(self, affixoid: Affixoid):
        return []
