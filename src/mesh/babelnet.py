from .utils import get_data_dir
import requests
import requests_cache
from hanziconv import HanziConv

class BabelNetAPI:
    def __init__(self):
        key_path = get_data_dir()/"babelnet_key.txt"
        if not key_path.exists():
            print("Please put your api key (provided by babelnet.org) in babelnet_key.txt")            
            raise FileNotFoundError()

        with key_path.open("r") as fin:
            self.bn_key = fin.read().strip()
        self.bn_url = "https://babelnet.io/v5/"
        requests_cache.install_cache()

    def http_get(self, url, params):
        headers = {"Accept-Encoding": "gzip"}
        return requests.get(url, params=params, headers=headers)

    def get_response(self, resp):
        if resp.status_code == 200:
            return resp.json()
        else:
            return {}

    def get_version(self):
        resp = self.http_get(self.bn_url+"getVersion", params={"key": self.bn_key})
        data = self.get_response(resp)
        return data.get("version")

    def get_senses(self, lemma, lang="ZH"):
        """Get senses from Babelnet
        
        Parameters
        ----------
        lemma: lemma to lookup  
        lang: lanugage code in BabelNet (e.g. EN, ZH)
              see https://babelnet.org/4.0/javadoc/it/uniroma1/lcl/jlt/util/Language.html
        """
        lemma = HanziConv.toSimplified(lemma)
        url = self.bn_url + "getSenses"
        resp = self.http_get(url, params=dict(
            lemma=lemma, key=self.bn_key,
            searchLang=lang, targetLang="EN"
        ))

        data = self.get_response(resp)
        return data
    
    def get_synset(self, synset_id):
        """Get Babelnet synset data
        
        Parameters
        ----------
        synset_id: synset id        
        """
        url = self.bn_url + "getSynset"
        resp = self.http_get(url, params=dict(
            id=synset_id, key=self.bn_key
        ))

        data = self.get_response(resp)
        return data

