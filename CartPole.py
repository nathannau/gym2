import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

print(tf.version.VERSION)

env = gym.make('CartPole-v1')

state = env.reset()
action = 1
done = False
while not done :
    env.render()
    _, _, done, _ = env.step( action )
    #if done : break
    action = 1-action

#input('Prout')
env.close()