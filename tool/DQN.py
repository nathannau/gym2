import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tool.Memory as Memory


class DQNAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            # optimizer=ko.Adam()
            # optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._value_loss]
        )

    def exploit(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            # action, _ = self.model.action_value(obs[None, :])
            action, _ = self.model.predict(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _value_loss(self, acts_and_advs, returns):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # actions = tf.one_hot(actions, env.action_space.n)
        # toto = actions.numpy()
        actions = tf.cast(actions, tf.int32)
        actions = tf.one_hot(actions, self.model.num_actions)

        advantages *= actions
        returns *= actions
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, advantages)

    def train(self, env, batch_sz=32, updates=1000):
        memory = Memory(50000)
        # # storage helpers for a single batch of data
        # actions = np.empty((batch_sz,), dtype=np.int32)
        # rewards, dones, values = np.empty((3, batch_sz))
        # observations = np.empty((batch_sz,) + env.observation_space.shape)
        # # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            if update % 50 == 0:
                print("{}/{} : {}".format(update, updates, ep_rews[-1]))
            for step in range(batch_sz):
                observation = next_obs.copy()
                # action, value = self.model.action_value(next_obs[None, :])
                value = self.model.predict(next_obs[None, :])
                action = np.argmax(value)
                next_obs, reward, done, _ = env.step(action)
                memory.append({
                    'observation': observation,
                    'action': action,
                    'value': value,
                    'reward': reward,
                    'done': done,
                    'next_obs': next_obs
                })
                if update % 5 == 0:
                    env.render()

                ep_rews[-1] += reward
                if done:
                    print(ep_rews[-1])
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            mini_batch = memory.reduce(batch_sz)
            mb_next_obs = mini_batch.next_obs
            next_reward = self.model.call(mb_next_obs)
            next_reward = tf.math.reduce_max(next_reward, axis=1)
            advs = self.params['gamma'] * next_reward * (1 - np.array(mini_batch.done))
            # target = tf.one_hot(mini_batch.action, env.action_space.n) * advantages
            acts_and_advs = np.concatenate([np.array(mini_batch.action)[:, None], advs[:, None]], axis=-1)
            losses = self.model.train_on_batch([mini_batch.observation], acts_and_advs)
            print(losses)

            # _, next_value = self.model.action_value(next_obs[None, :])
            # returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            # acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            # losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages


class DQNModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQNModel, self).__init__()
        self.num_actions = num_actions
        self.hidden1 = kl.Dense(128, activation='relu')
        self.logits = kl.Dense(num_actions, name='policy_logits')  # , activation='softmax')
        # self.softmax = kl.Softmax(num_actions, name='policy_logits')
        # self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        hidden_logs = self.hidden1(x)
        return self.logits(hidden_logs)

    def action_value(self, obs):
        # executes call() under the hood
        # value = tf.constant([[0.5, 0.5]])
        # print(value)
        # action = tf.keras.activations.softmax(value, axis=0)
        # print(value)
        value = self.predict(obs)
        # print(value)
        # value = np.squeeze(value, axis=0)
        # print(value)
        # action = tf.keras.activations.softmax(value, axis=0)
        # print(value)
        # return (logits, np.squeeze(action, axis=-1), np.squeeze(value, axis=-1 ))
        # return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
        return np.squeeze(value, axis=0)
