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
# model = tool.A2CModel(num_actions=env.action_space.n)
# if input("Reload model : ") == "y":
#     model.load_weights('truc_w.tf')

agent = tool.DQNAgent(model)
# agent = tool.A2CAgent(model)

rewards_history = agent.train(env, batch_sz=100, updates=600, see_each=5)
model.save_weights('truc_w.tf')
# tf.keras.models.save_model(model, 'truc.tf')
print("Finished training, testing...")
print("%d out of 200" % agent.exploit(env, True))
# agent.exploit(env, True)
