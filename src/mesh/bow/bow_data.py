from dataclasses import dataclass
from ..utils import get_data_dir
from nltk.corpus import WordNetCorpusReader

@dataclass
class BowItem:
    wn_offset: str    
    meaning: str
    examples: str
    translation: str
    lemmas: str
    head_word: str
    pos: str
    SUMO: str
    SUMO_zhTw: str
    MILO: str
    MILO_zhTw: str
    synset: str

    @staticmethod
    def from_list(data_list, wn):
        item = BowItem(*[*data_list, ""])        
        item.translation = [x.strip() for x in item.translation.split("＠")]
        item.MILO_zhTw = item.MILO_zhTw.strip()
        item.examples = item.examples.replace("\"", "")

        try:
            offset = int(item.wn_offset[:-1])
            pos = item.wn_offset[-1]
            item.synset = wn.synset_from_pos_and_offset(pos.lower(), offset).name()
        except Exception as ex:        
            pass
        return item

def load_BOW(wn):
    bow_path = get_data_dir() / "bow/BOW.txt"
    with bow_path.open("r", encoding="UTF-8") as fin:
        fin.readline()
        bow = [BowItem.from_list(x.split("\t"), wn) for x in fin.readlines()]
    return bow