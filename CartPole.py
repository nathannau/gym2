import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from gym.envs.classic_control import CartPoleEnv
import tool

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['AUTOGRAPH_VERBOSITY'] = '10'

print(tf.version.VERSION)

env = gym.make('CartPole-v1')

model = tool.DQNModel(num_actions=env.action_space.n)

obs = env.reset()
# print("obs, obs[None, :]")
# print(obs, obs[None, :])
# a = model.predict(obs[None, :])
action, value = model.action_value(obs[None, :])
# print(a)
# print("action, value")
print(action, value)
exit()
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.shape)
# print(env.observation_space.low)
# print(env.observation_space.high)

# print(env.observation_space[2])
# print(env.observation_space[3])

env._max_episode_steps = 100
state = env.reset()

action = 1
done = False
while not done:
    env.render()

    if state[2] > 0.10:
        action = 1
    elif state[2] < -0.10:
        action = 0
    elif state[3] > 0.50:
        action = 1
    elif state[3] < -0.50:
        action = 0
    elif state[0] > 1:
        action = 1
    elif state[0] < -1:
        action = 0
    elif state[1] > 1:
        action = 1
    elif state[1] < -1:
        action = 0
    else:
        action = 1-action

    print(state, action)

    state, _, done, _ = env.step(action)

# input('Prout')
env.close()
