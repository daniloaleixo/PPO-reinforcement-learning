import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import gym
import datetime as dt

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n

class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        value, logits = self.predict_on_batch(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value


model = Model(num_actions)
model.load_weights("saved_models/PPO")

# Reset episode
num_test_int = 10
global_info = []

for i in range(num_test_int):
    time, cumul_reward, done = 0, 0, False
    old_state = env.reset()
    
    while not done:
        env.render()
        # Actor picks an action (following the policy)
        action, value = model.action_value(old_state.reshape(1, -1))
        # Retrieve new state, reward, and whether the state is terminal
        new_state, reward, done, _ = env.step(action.numpy()[0])
        # Update current state
        old_state = new_state
        cumul_reward += reward
        time += 1
        
        if done: 
            global_info.append({
                cumul_reward
            })