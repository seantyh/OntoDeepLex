from nltk.corpus import wordnet as wn

class SynsetDescriptor:
    def __init__(self, offset, pos, lemmas, defs, examples):
        self.offset = offset
        self.pos = pos
        self.lemmas = lemmas
        self.definition = defs
        self.examples = examples

    def __repr__(self):
        return f"<SynsetDescriptor: {self.offset}{self.pos}: {self.definition}>"

def get_synset(syn_id):
    syn_pos = syn_id[-1]
    syn_num = syn_id[:-1]
    return wn.synset_from_pos_and_offset(syn_pos, int(syn_num))

def get_pwn_relations(syn_id, relations=None, depth=1):
    if isinstance(syn_id, str):
        syn_x = get_synset(syn_id)
    else:
        syn_x = syn_id
    
    RELATIONS = [
        "hypernyms", "hyponyms",
        "member_holonyms", "member_meronyms",
        "part_holonyms", "part_meronyms",
        "substance_holonyms", "substance_meronyms"
    ]

    if relations:
        if isinstance(relations, str):
            relations = [relations]
        if all(rel_x in RELATIONS for rel_x in relations):
            pass
        else:
            raise ValueError("some relations are not valid in " + str(relations))
    else:    
        relations = RELATIONS

    neighbors = []
    
    for rel_x in relations:        
        buf = [(syn_x, 0)]
        while buf:
            syn_seed, gen_i = buf.pop()
            if depth > 0 and gen_i >= depth:
                break            
            rel_method = getattr(syn_seed, rel_x)
            rel_seeds = rel_method()
            neighbors.extend([
                (syn_y, rel_x, gen_i) for syn_y in rel_seeds
            ])
            buf.extend([(x, gen_i+1) for x in rel_seeds])

    return neighbors