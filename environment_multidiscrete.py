from cmath import log10
from time import sleep
import gym
import numpy as np
from gym import spaces
from graph import TAP

from graphvisual import GraphVisualization


# This defines the environment that uses the multi-discrete action space

class TAPEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,n_verts,density):
        super(TAPEnv, self).__init__()
        self.n_verts = n_verts
        self.density = density

        
        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([n_verts, n_verts])
        self.observation_space = spaces.Box(low=False, high=True,
                                            shape=(2, n_verts, n_verts), dtype=bool)

    # function that returns the current observation space
    def get_obs(self):
        return np.array([self.tap.get_graph(), self.tap.get_tree()], dtype=bool)

    # takes an action and advances the environment
    def step(self, action):
        u = action[0]
        v = action[1]
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
        G = GraphVisualization()

        for i in range(self.n_verts):
          for j in range(self.n_verts):
              if self.tap.get_graph()[i,j]:
                  G.addEdge(i,j)
        G.visualize()

    # masks the current action space.
    # this function does not work, this is the multi-discrete space
    # this could fixed, but it likely wouldnt perform as well as
    # the discrete action space due to limitations on multi-discrete action masking
    def action_masks(self):
        l = []

        g = self.tap.get_graph()
        q = 0

        for i in range(self.n_verts):
            q = q+1
            for j in range(q, self.n_verts):
                if g[i,j]:
                  l.append([i, j])
        return l

    # unused
    def close (self):
        return