import igraph as ig
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

def compute_average_distance(G: ig.Graph):
    psum = 0
    pcount = 0
    for v in tqdm(G.vs, desc="calc. avg. distance"):
        us = list(range(v.index))
        plen = [x for x in G.shortest_paths(v, us)[0] if x != float('inf')]
        psum += sum(plen)
        pcount += len(plen)

    if pcount == 0:
        return np.nan
    else:
        return psum/pcount

def compute_struct_indices(G: ig.Graph):
    if G.ecount() == 0:
        print("no connection exists")
        return {}

    G = G.copy().simplify()
    deg = G.degree()    
    cliques = G.cliques(min=3)
    compos = G.components()
    avg_distance = compute_average_distance(G)
    res = pd.Series({
        "nV": G.vcount(), 
        "nE": G.ecount(),
        "Avg Degree": np.mean(deg),
        "Max Degree": np.max(deg),
        "Diameter": G.diameter(), 
        "Avg distance": avg_distance,
        "Global clustering coeff.": G.transitivity_undirected(),
        "Avg local clustering coeff.": np.nanmean(G.transitivity_local_undirected()),
        "Degree assortativity": G.assortativity_degree(),
        "Largest clique size": max(len(x) for x in cliques),
        "Isolated nodes": sum(1 for x in compos if len(x) == 1),
        "Largest component": max(len(x) for x in compos)
    })
    return res