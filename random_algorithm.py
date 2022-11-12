from math import log2
from graph import TAP
import random

# this file evaluates the performance of randomly selecting valid edges

#set properties of the graph
n_verts = 100
density = .05

samples = 100
count = 0
for _ in range(samples):
    tap = TAP(n_verts)
    tap.randomize(density)
    while not tap.no_edges():
        
        u = random.randrange(n_verts)
        v = random.randrange(n_verts)
        g = tap.get_graph()
        while not g[u,v]:
            if u == n_verts-1:
                v = (v+1) % n_verts
            u = (u+1) % n_verts

        tap.merge_path(u,v)
        count = count + 1

print(f"avg: {count/samples}")
