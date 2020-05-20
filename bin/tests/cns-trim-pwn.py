from mesh_import import mesh
import igraph as ig
import pytest

def test_trim():
    testG = ig.Graph()
    testG.add_vertices(10)
    testG.add_edges([[0,1],[0,2],[0,3],[0,4],[4,5], [4,6], [4,7], [7, 8], [7,9]])
    layout = testG.layout("kk")
    ig.plot(testG, layout=layout)
