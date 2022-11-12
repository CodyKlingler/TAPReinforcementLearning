import gym
import numpy as np
from gym import spaces
from graph import TAP
from typing import List
from graphvisual import GraphVisualization

# environment for discrete action space
# used for masked and unmasked algorithms

class DiscTAPEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,n_verts,density):
        super(DiscTAPEnv, self).__init__()
        self.n_verts = n_verts
        self.density = density
        # use only one half of the action space to reduce redundancy
        self.n_actions = (n_verts*(n_verts-1))//2 

        # array to turn an action number into a tuple
        self.action_to_vert = []
        q = 0
        for r in range(0, self.n_actions-1):
            q = q+1
            for c in range(q, n_verts):
                self.action_to_vert.append((r, c))

        # function that returns the current observation space
        self.action_space = spaces.Discrete(self.n_actions) # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=False, high=True,
                                            shape=(2, n_verts, n_verts), dtype=bool)

    # only used if the environment is passed to a masked algorithm
    # returns true for the action indices that are valid
    def action_masks(self) -> List[bool]:
        l = []

        g = self.tap.get_graph()

        for i in range(0,self.n_actions):
            (u,v) = self.action_to_vert[i]
            l.append(g[u,v])
        return l

    # function that returns the current observation space
    def get_obs(self):
        return np.array([self.tap.get_graph(), self.tap.get_tree()], dtype=bool)

    # takes an action and advances the environment
    def step(self, action):
        (u, v) = self.action_to_vert[action]
        
        reward = -1

        self.tap.merge_path(u, v)

        observation = self.get_obs()
        done = self.tap.no_edges()
        info = {}
        return observation, reward, done, info

    # sets the environment to a random initializition
    # creates a Tree Augmentation object
    def reset(self):
        self.tap = TAP(self.n_verts)
        self.tap.randomize(self.density)
        observation = self.get_obs()
        return observation # reward, done, info can't be included

    # uses matplotlib to show the current graph
    def render(self):

        # Driver code
        G = GraphVisualization()

        for i in range(self.n_verts):
          for j in range(self.n_verts):
              if self.tap.get_graph()[i,j]:
                  G.addEdge(i,j)
        G.visualize()

    # unused
    def close (self):
        return