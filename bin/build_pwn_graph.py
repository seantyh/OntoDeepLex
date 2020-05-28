from mesh_import import mesh
import igraph as ig
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

cns_dir = mesh.get_data_dir() / "cns"
G_pn = mesh.build_pwn_igraph(pos='n', included_rel=["hypernyms", "holonyms"])
print(G_pn.summary())

Gpn_prop = mesh.compute_struct_indices(G_pn)
print(Gpn_prop)

G_pn.write(cns_dir / "cwn_sense_graph_pn.pkl", format="pickle")

# -- Verb synsets ---
G_pv = mesh.build_pwn_igraph(pos='v', included_rel=["hypernyms", "holonyms"])
print(G_pv.summary())

Gpv_prop = mesh.compute_struct_indices(G_pv)
print(Gpv_prop)

G_pv.write(cns_dir / "cwn_sense_graph_pv.pkl", format="pickle")
