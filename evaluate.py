import numpy as np
import gym
import datetime as dt
from PPO.model import Model

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n

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