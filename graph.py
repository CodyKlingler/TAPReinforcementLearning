import numpy as np
import queue
import random

# this file defines the logic for graphs and trees
# also defines the logic for the TAP (tree augmentation problem) class

# handles things like merging tree paths and random initialization

class Graph: 

    def __init__(self, n_verts):
        self.is_tree = False
        self.rt = 0
        self.n_verts = n_verts
        self.n_edges = 0
        self.adj = np.full((n_verts,n_verts), False, dtype=bool)
        self.retained = queue.Queue()
        self.identity = np.array(range(0, n_verts), dtype=np.uint8)

    def add_edge(self,u,v):
        u = self.deref(u)
        v = self.deref(v)

        if u == v:
            return

        if not self.adj[u,v]:
            self.n_edges = self.n_edges + 1

        self.adj[u,v] = True
        self.adj[v,u] = True


        if self.is_tree:
            self.set_root(self.root()) ################### SLOW

    # used only for trees
    def set_root(self, v):
        v = self.deref(v)
        self.is_tree = True
        self.rt = v
        self.parents = np.zeros(self.n_verts, dtype=np.uint8)

        q = queue.Queue()
        q.put(self.rt)

        processed = np.full(self.n_verts, False,dtype=bool)
        processed[self.rt] = True

        while q.qsize() > 0:
            cur_v = q.get()
            neighbors = self.adj[cur_v,:]
            for i in range(self.n_verts):
                if neighbors[i] and not processed[i]:
                    self.parents[i] = cur_v
                    q.put(i)
                    processed[i] = True

    # returns root
    def root(self):
        return self.deref(self.rt)

    # returns the new value of a merged vertex
    def deref(self,v):
        u = v
        while u != self.identity[u]:
            u = self.identity[u]
        self.identity[v] = u
        return u

    # only works for trees
    def parent(self,u):
        u = self.deref(u)
        return self.deref(self.parents[u])

    def merge(self, u, v):
        self.merge_no_search(u,v)

        if self.is_tree:
            self.set_root(min(u,v))
            '''if self.deref(self.parents[u]) == self.deref(u):
                self.parents[u] = self.deref(self.parents[v])
            elif self.deref(self.parents[v]) == self.deref(v):
                self.parents[v] = self.deref(self.parents[u])
            '''

    # merging without replacing the root
    def merge_no_search(self,u,v):
        u = self.deref(u)
        v = self.deref(v)
        mu = min(u,v)
        mv = max(u,v)

        if mu == mv:
            return

        self.identity[mv] = mu

        row_adjv = self.adj[mv,:]
        col_adjv = self.adj[:,mv]

        removed = np.logical_and(row_adjv, self.adj[mu,:])
        n_removed = 0
        for i in range(self.n_verts):
            if removed[i]:
                n_removed = n_removed+1

        self.adj[mu,:] = np.logical_or(row_adjv, self.adj[mu,:])
        self.adj[:,mu] = np.logical_or(col_adjv, self.adj[:,mu])

        self.adj[mv,:] = np.full((self.n_verts,),False, dtype=bool)
        self.adj[:,mv] = np.full((1,self.n_verts),False, dtype=bool)

        if self.adj[mu,mu]:
            self.adj[mu,mu] = False
            n_removed = n_removed+1

        self.n_edges = self.n_edges - n_removed

    def no_edges(self):
        return self.n_edges == 0
            
    def depth(self,u):
        rt = self.root()
        depth = 0
        while self.parent(u) != rt:
            u = self.parent(u)
            depth = depth +1
        return depth

    def tree_path(self,u,v):
        if not self.is_tree:
            raise AttributeError(f"Can't find path ({u},{v}) becuase graph is not a tree. Try setting a root with set_root()")
        u = self.deref(u)
        v = self.deref(v)

        path = list()

        d_u = self.depth(u)
        d_v = self.depth(v)

        qu = []
        qv = []

        while u != v:
            if d_u > d_v:
                qu.append(u)
                u = self.parent(u)
                d_u = d_u-1
            else:
                qv.append(v)
                v = self.parent(v)
                d_v = d_v-1

        qu.append(u)
        while len(qv) > 0:
            qu.append(qv.pop())

        return qu

    def merge_path(self,vs):
        for i in range(len(vs)-1):
            self.merge_no_search(vs[i], vs[i+1])
        
        if self.is_tree:
            self.set_root(self.rt)

    def random_tree(self):
        is_added = np.full((self.n_verts,), False)

        added = []
        not_added = range(self.n_verts)
        
        first = random.randrange(self.n_verts)
        added.append(first)

        is_added[first] = True

        for i in range(self.n_verts-1):
            new_v = random.randrange(self.n_verts)
            while is_added[new_v]:
                new_v = (new_v + 1) % self.n_verts
            
            is_added[new_v] = True
            rand_t = random.randrange(len(added))
            self.add_edge(new_v, rand_t)
            added.append(new_v)

        self.set_root(0)
    
    def random_graph(self,density):             ############## not strict size

        if density > 1.0 or density < 0:
            print("density error")
        
        complete_edges = (self.n_verts*(self.n_verts-1))//2
        n_edges = int((density*complete_edges)+.5)

        for i in range(n_edges):
            u = random.randrange(self.n_verts)
            v = random.randrange(self.n_verts)
            while self.adj[u,v]:
                if u == self.n_verts-1:
                    v = (v+1)%self.n_verts
                u = (u+1)%self.n_verts

            self.add_edge(u,v)

    def print_adj(self):
        print(self.adj)

    def print(self):
        for i in range(self.n_verts):
          for j in range(self.n_verts):
              if self.adj[i,j]:
                  print(i,j)


class TAP:
    def __init__(self,n_verts):
        random.seed()
        self.n_verts = n_verts
        self.tree = Graph(n_verts)
        self.graph = Graph(n_verts)

    def no_edges(self):
        return self.graph.no_edges()

    def set_root(self, u):
        self.tree.set_root(u)

    def randomize(self, density):
        self.tree.random_tree()
        self.graph.random_graph(density)

    def add_tree(self, u, v):
        self.tree.add_edge(u,v)
    
    def add_graph(self, u, v):
        self.graph.add_edge(u,v)

    def merge_path(self, u, v):
        vs = self.tree.tree_path(u,v)
        self.tree.merge_path(vs)
        self.graph.merge_path(vs)

    def get_graph(self):
        return self.graph.adj

    def get_tree(self):
        return self.tree.adj
