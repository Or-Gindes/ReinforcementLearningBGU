import gymnasium as gym
import numpy as np
import pygame

def getMaxQ_Value(state_s_t):
    # Get the row at index state_s_t - all q values with selected state all over the actions
    row_i = lookup_table[state_s_t, :]

    # Get the maximum value in the row
    max_value = np.max(row_i)
    return max_value


env =gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,render_mode = 'human') # Init a MDP environments
observation, info = env.reset() # Resets the environment to an initial state,
lookup_table = np.zeros((env.observation_space.n,env.action_space.n))
hyper_parameters = {"learning_rate" :0,"discount_factor" :0.5,"decay_rate" :0 }
init_state = True

for _ in range(1000):
    if init_state:
        s =observation
        init_state = False
    else:
        s = state_s_prime
    action = env.action_space.sample()  # agent policy that uses the observation and info
    state_s_prime, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        lookup_table[s][action] = reward
        observation, info = env.reset()
    else:
        lookup_table[s][action] = reward + hyper_parameters["discount_factor"]* getMaxQ_Value(state_s_prime)

env.close()

