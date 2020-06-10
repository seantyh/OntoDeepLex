import shelve
from CwnGraph import CwnBase
from ..utils import get_data_dir, ensure_dir
from .sense_model_alpha import WsdBaselineBuilder, WsdBaseline
from tqdm.autonotebook import tqdm
from functools import lru_cache

class SenseTagger:
    def __init__(self):
        senses_dir = get_data_dir()/"senses"
        ensure_dir(senses_dir)
        self.wsd_units = shelve.DbfilenameShelf(
                        str(senses_dir/'wsd.units'), "c", 
                        writeback=True)
        self.cwn_cache = {}
        self.cwn = CwnBase()

    def tag(self, text):
        if isinstance(text, str):
            raise ValueError("expect text to be a presegmented word list")
        words = set(text)
        results = []
        for word in tqdm(words):
            wsd = self.get_wsd_unit(word)
            if not wsd:
                continue
            results.extend(wsd.predict(text))
        self.wsd_units.sync()

        ret_list = [None] * len(text)
        for idx, sense_x in results:
            ret_list[idx] = sense_x
        return ret_list

    def is_in_cwn(self, word):        
        if word not in self.cwn_cache:
            self.cwn_cache[word] = bool(self.cwn.find_all_senses(word))
        return self.cwn_cache[word]

    @lru_cache(maxsize=500)
    def get_wsd_unit(self, word):
        if not self.is_in_cwn(word):
            return None

        if word not in self.wsd_units:        
            wsd = WsdBaselineBuilder().build(word)
            self.wsd_units[word] = wsd.knn
        else:
            knn = self.wsd_units[word]
            wsd = WsdBaseline(word, knn)
        return wsd
        