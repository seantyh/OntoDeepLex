import igraph as ig
from nltk.corpus import wordnet as wn
from tqdm.autonotebook import tqdm

PWN_RELATIONS = [
        "hypernyms", "hyponyms",
        "member_holonyms", "member_meronyms",
        "part_holonyms", "part_meronyms",
        "substance_holonyms", "substance_meronyms"
    ]

def add_vertex(vertex_name, G: ig.Graph, **kwargs):
    try:
        v = G.vs.find(name=vertex_name)
    except ValueError:
        v = G.add_vertex(vertex_name, **kwargs)
    return v

def add_relation(xid: str, yid: str, G: ig.Graph, **kwargs):
    try:
        e = G.es.find(_source=xid, _target=yid)
    except ValueError as ex:
        # no such vertex or edge
        e = None

    if not e:
        try:
            e = G.add_edge(xid, yid, **kwargs)
        except ValueError as ex:
            # no such vertex
            e = None
    return e

def normalize_relations(relations):
    import re
    if relations is None:
        return PWN_RELATIONS

    norm_rels = set()

    for r in relations:
        norm_rels.update(x for x in PWN_RELATIONS if re.search(r, x, re.IGNORECASE))
    return list(norm_rels)

def build_pwn_igraph(**kwargs):    

    directed = kwargs.get("directed", False)
    pos = kwargs.get('pos', None)
    G = ig.Graph(directed=directed)
    for syn_x in tqdm(wn.all_synsets(pos), desc='adding vertices'):
        G.add_vertex(syn_x.name())

    #pylint: disable=all
    G.vs_names = set(G.vs["name"])

    included_rel = kwargs.get("included_rel", None)
    included_norm_rel = normalize_relations(included_rel)
    print("included rel_type: %s" % str(included_norm_rel))

    for syn_x in tqdm(wn.all_synsets(pos), desc='adding edges'):
        for rel_type in included_norm_rel:
            rel_method = getattr(syn_x, rel_type)
            rel_synsets = rel_method()
            for rel_x in rel_synsets:
                add_relation(syn_x.name(), rel_x.name(), G, rel_type=rel_type)

    return G