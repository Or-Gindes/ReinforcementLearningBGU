import gymnasium as gym
import numpy as np
import pygame
import matplotlib.pyplot as plt





episodes_bench_marks = {500,2000,5000,5000}



def train_agent(num_episodes =5000,max_step_num=100):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
                   #render_mode='human')  # Init a MDP environments
    state_s, info = env.reset()  # Resets the environment to an initial state,
    lookup_table = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.8
    gamma = 0.8
    epsilon = 0.6
    epsilon_decay_rate = epsilon / num_episodes # epsilon decay rate. 1/0.0001 = 10,000


    rewards_per_episode = []  # Track rewards per episode
    steps_to_goal = []  # Track number of steps to goal per episode
    final_to_goal = []
    final_idx = []



    for episode_idx in range(1,num_episodes+1):
        total_reward = 0
        num_steps = 0

        terminated = False
        truncated = False

        #for _ in range(max_step_num):
        while (not terminated and not truncated):
            num_steps += 1

            action = sample_action(env, epsilon, lookup_table, state_s)

            state_s_prime, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                target = reward
                state_s_prime, info = env.reset()
            else:
                target = reward + gamma* np.max(lookup_table[state_s_prime, :])
            lookup_table[state_s][action] = (1-learning_rate)*lookup_table[state_s][action] + learning_rate*target
            state_s = state_s_prime
        if truncated:
            num_steps = 100

        steps_to_goal.append(num_steps)


        if episode_idx % 100 == 0:
            if len(steps_to_goal) >= 100:
                avg_steps = np.mean(steps_to_goal[-100:])
                final_to_goal.append(avg_steps)
                final_idx.append(episode_idx)


        if episode_idx in episodes_bench_marks:
            plot_heat_map(episode_idx, lookup_table)
        epsilon = max(epsilon - epsilon_decay_rate,0)
        rewards_per_episode.append(total_reward)
    plot_rewards_per_episode(rewards_per_episode)
    plot_avg_steps_per_100_episodes(final_idx,final_to_goal)


    env.close()


def sample_action(env, epsilon, lookup_table,  state_s):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # agent policy that uses the observation and info
    else:
        action = np.argmax(lookup_table[state_s, :])
    return action


def plot_heat_map(_, lookup_table):
    plt.imshow(lookup_table, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar to show the mapping of values to colors

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(lookup_table.shape[1]), np.arange(lookup_table.shape[0]))

    # Annotate the plot with each value in the array
    for x, y, value in zip(x_coords.flatten(), y_coords.flatten(), lookup_table.flatten()):
        plt.text(x, y, "{:.2f}".format(value), ha='center', va='center', color='white', fontsize=6)
        # Set custom x-axis tick labels
    plt.xticks(np.arange(lookup_table.shape[1]), ['left', 'down', 'right', 'up'], fontsize=6)
    plt.text(lookup_table.shape[1] / 2, -2, f"number of episodes = {_}",
             ha="center", va="center", fontsize=8)
    plt.savefig(f'heat_map_{_}.jpg', format='jpeg')  # Adjust quality as needed
    plt.show()


def plot_rewards_per_episode(rewards_per_episode):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid()
    plt.savefig('rewards_per_episode.jpg', format='jpeg')  # Adjust quality as needed

    plt.show()


def plot_avg_steps_per_100_episodes(episode_idx, avg_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_idx, avg_steps, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps to Goal (Last 100 Episodes)')
    plt.title('Average Steps to Goal over Last 100 Episodes')
    plt.grid()
    plt.savefig('avg_steps_per_100_episodes.jpg', format='jpeg')  # Adjust quality as needed

    plt.show()



if __name__ == '__main__':
    train_agent(5000)
