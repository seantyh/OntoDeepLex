from typing import List
import igraph as ig
from tqdm.autonotebook import tqdm
from CwnGraph import CwnBase, CwnSense, CwnFacet
import logging

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

def vertex_compo_size(G: ig.Graph):
    compo_list = G.components()
    compo_map = {}
    for compo_x in compo_list:
        n_compo_x = len(compo_x)
        compo_map.update({
            vidx: n_compo_x for vidx in compo_x
        })
    v_compo_size = [compo_map.get(vidx, 0) for vidx in range(G.vcount())]
    return v_compo_size

def build_igraph(cwn, **kwargs):
    logger = logging.getLogger("build_igraph")
    logger.setLevel("INFO")

    directed = kwargs.get("directed", False)
    G = ig.Graph(directed=directed)
    for sense_x in cwn.senses():
        G.add_vertex(sense_x.id)

    #pylint: disable=all
    G.vs_names = set(G.vs["name"])

    included_rel = kwargs.get("included_rel", None)
    use_pwn = kwargs.get("use_pwn", False)

    logger.info("included rel_type: %s", str(included_rel))

    for sense_x in tqdm(cwn.senses()):
        for rel_type, sense_y, rel_dir in sense_x.semantic_relations:
            if rel_dir == "reversed":
                continue
            if included_rel is not None \
                and rel_type not in included_rel:
                continue
            if isinstance(sense_y, CwnFacet):
                add_relation(sense_x.id, sense_y.sense.id, G, rel_type=rel_type)
            elif isinstance(sense_y, CwnSense):
                add_relation(sense_x.id, sense_y.id, G, rel_type=rel_type)
            else:
                pass

        if use_pwn:
            pwn_syns = sense_x.pwn_synsets
            for rel_type, syn_x in pwn_syns:            
                if not syn_x.has_wn30:
                    continue
                wn_syn = syn_x.wn30_synset
                pwn_id = wn_syn.name()
                add_vertex(pwn_id, G, pwn='pwn')                    
                G.add_edge(sense_x.id, pwn_id, rel_type="align_" + rel_type)
                G.add_edge(pwn_id, sense_x.id, rel_type="align_" + rel_type)

                # add nodes along hypernym paths                                
                add_hypernym_paths(G, syn_x.wn30_synset.hypernym_paths())                

    return G

def add_hypernym_paths(G, hyper_paths):
    # from https://docs.python.org/3/library/itertools.html
    def pairwise(iterable):
        from itertools import tee
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    for hyp_path in hyper_paths:
        get_synid = lambda x: x.name()
        for x, y in pairwise(hyp_path):
            xid, yid = get_synid(x), get_synid(y)
            add_vertex(xid, G, pwn="pwn")
            add_vertex(yid, G, pwn="pwn")                        
            add_relation(yid, xid, G, rel_type="hypernym", pwn="pwn")        

def compo_size_distribution(G: ig.Graph, **kwargs):
    from collections import Counter
    import matplotlib.pyplot as plt
    compos = list(G.components())

    # compo_sizes: Dict[CompoSize, Count]
    compo_sizes = Counter(len(x) for x in compos)
    compo_distr = sorted(list(compo_sizes.items()), key=lambda x: x[0])

    title_label = kwargs.get("title_label", "")

    plt.plot([x[0] for x in compo_distr], [x[1] for x in compo_distr], '.')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title_label + " - Component size distribution")
    ax.set_xlabel("Component size")
    ax.set_ylabel("Count")

    return compo_distr

def trim_pwn(G):
    from itertools import combinations, product
    import time
    cwn_vertices = list(G.vs.select(pwn=None))
    pwn_vertices = list(G.vs.select(pwn="pwn"))
    edges_to_add = set()    

    # find connected CWNs
    for pwn_v in tqdm(pwn_vertices):
        edges = find_conn_cwn_with_pwn(G, pwn_v, cwn_vertices)        
        edges_to_add.update(edges)                        

    # add implied edges in graph    
    for u, v in tqdm(edges_to_add, desc="adding edges"):
        G.add_edge(u, v, rel_type="pwn_implied")

    
    # trim PWN vertices
    G.delete_vertices(G.vs.select(pwn="pwn"))        

def find_conn_cwn_with_pwn(G: ig.Graph, 
        pwn_vertex: ig.Vertex, cwn_vertices: List[ig.Vertex]):
    src_vertices = []
    tgt_vertices = []

    inf = float('inf')
    # from pwn_vertex to cwn_vertices
    out_paths = G.shortest_paths(pwn_vertex, cwn_vertices, mode=ig.OUT)[0]
    for pi, path_len in enumerate(out_paths):
        cwn_v = cwn_vertices[pi]
        if path_len != inf:
            tgt_vertices.append(cwn_v)
    
    # from cwn_verteices to pwn_vertices
    in_paths = G.shortest_paths(pwn_vertex, cwn_vertices, mode=ig.IN)[0]
    for pi, path_len in enumerate(in_paths):
        cwn_v = cwn_vertices[pi]
        if path_len != inf:
            src_vertices.append(cwn_v)
    
    cwn_paths = G.shortest_paths(src_vertices, tgt_vertices, mode=ig.OUT)
    edges_to_add = []
    for src_i, src_v in enumerate(src_vertices):
        for tgt_i, tgt_v in enumerate(tgt_vertices):
            cwn_path_len = cwn_paths[src_i][tgt_i]            
            if cwn_path_len > 1:
                edges_to_add.append((src_v, tgt_v))

    return edges_to_add


def trim_pwn_in_component(G, vertices):
    from itertools import combinations, product
    from tqdm.autonotebook import tqdm

    cwn_vertices = [x for x in vertices if not G.vs["pwn"][x]]
    if len(cwn_vertices) > 1000:
        cwn_iter = tqdm(cwn_vertices)
    else:
        cwn_iter = iter(cwn_vertices)

    for u in cwn_iter: 
        paths = G.get_shortest_paths(u, to=cwn_vertices)
        # print("Start from vertex: %d" % (u.index,))
        # print(paths)
        for path_i, path in enumerate(paths):
            v = cwn_vertices[path_i]
            if u == v: continue            
            if path:
                has_pwn = any(G.vs["pwn"][x] for x in path)                
                if has_pwn:
                    G.add_edge(u, v, rel_type="pwn_implied")
            else:
                pass                

def draw_igraph(G: ig.Graph):
    import networkx as nx
    if G.is_directed():
        nG = nx.DiGraph()
    else:
        nG = nx.Graph()
    nG.add_nodes_from([x.index for x in G.vs])
    nG.add_edges_from([(x.source, x.target) for x in G.es])
    # layout = nx.kamada_kawai_layout(nG)
    layout = nx.spring_layout(nG)
    node_colors = ["red" if x.attributes().get("pwn") else "blue" for x in G.vs]
    nx.draw(nG, pos=layout, node_color=node_colors)
    nx.draw_networkx_labels(nG, pos=layout, font_color="white")
