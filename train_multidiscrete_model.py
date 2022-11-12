import gym
import numpy as np
from stable_baselines3 import PPO
import os
from environment_multidiscrete import TAPEnv
from environment_discrete import DiscTAPEnv

#set properties of the graph
n_verts =      100
density =      .05

#define model and environment type
model_type_s = "PPO_MultiDiscrete" # name used for generated logs and model files
model_type =   PPO
environment_type =       TAPEnv

# make true to load the model saved with name model_n
load = False
model_n = "/250000"

# create folders if needed
model_type_s = model_type_s + f"_({n_verts}, {density})"
models_dir = "models/" + model_type_s
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


# create training environment
env = environment_type(n_verts=n_verts, density=density)
env.reset()

# MlpPolicy = Multi-layer perceptron
model = model_type("MlpPolicy", env, verbose=1, tensorboard_log=logdir) #tell model what environment it learns from

# load model from file if necessary
if load:
    model_path = models_dir + model_n
    model = model_type.load(model_path, env=env)

# continually train the agent
TIMESTEPS = 10000
TRAINING_ITERATIONS = 100    # make this higher (>1000 maybe even >1,000,000) to train the agent for longer
for i in range(1,TRAINING_ITERATIONS+1): 
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_type_s+"_log")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# uncomment this to visualize the trained model
# (not very useful in my opinion)
''' 
episodes = 1
for ep in range(episodes):
    obs = env.reset()
    done = False
    i = 0
    while not done:
        i = i+1
        env.render()
        print(f"round: {i}")
        action, states = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()
'''