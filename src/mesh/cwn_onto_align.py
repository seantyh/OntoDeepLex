from dataclasses import dataclass, field
from CwnGraph import CwnSense
from nltk.corpus import wordnet as wn
from typing import List
from .pwn import get_pwn_relations, get_synset

@dataclass
class AlignmentData:
    zh_lemma = ""
    pwn_id = ""
    sense_list: List[CwnSense] = field(default_factory=list)
    related_synsets: List["Synset"] = field(default_factory=list)

    def __repr__(self):
        return "<AlignmentData: {zh_lemma}-{pwn_id}: "\
               "{n_sense} CWN senses, {n_rel_synsets} related PWN synsets>"\
            .format(**(dict(**self.__dict__, 
                n_sense=len(self.sense_list), 
                n_rel_synsets=len(self.related_synsets))))
    
    def to_dict(self):
        def sense_to_dict(sense):
            return dict(id=sense.id,
                pos=sense.pos,
                definition=sense.definition,
                examples=sense.all_examples())

        def synset_to_dict(syn):
            return dict(
                name=syn.name(),
                offset=syn.offset(),
                pos=syn.pos(),
                definition=syn.definition(),
                examples=syn.examples())

        out = dict(
            zh_lemma=self.zh_lemma,
            pwn_synset=synset_to_dict(get_synset(self.pwn_id)),
            sense_list=[sense_to_dict(x) for x in self.sense_list],
            related_synsets=[[synset_to_dict(x[0]), x[1], x[2]] 
                for x in self.related_synsets]
        )

        return out

    
def get_alignment_structure(zh_lemma, pwn_id, cwn, n=2):    
    sense_list = cwn.find_all_senses(zh_lemma)
    related_synsets = get_pwn_relations(pwn_id, depth=n)
    
    align_data = AlignmentData()    
    align_data.pwn_id = pwn_id
    align_data.zh_lemma = zh_lemma
    align_data.sense_list = sense_list
    align_data.related_synsets = related_synsets

    return align_data
