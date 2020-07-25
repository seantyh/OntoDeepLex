from ..utils import get_data_dir
import nltk
from nltk.corpus import WordNetCorpusReader

def get_wordnet16():
    wn_dir = str(get_data_dir() / "bow/wn16_dict")
    wn = WordNetCorpusReader(wn_dir, nltk.data.find(wn_dir))
    return wn