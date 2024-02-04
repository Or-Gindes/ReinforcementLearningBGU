import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

EPISODES_BENCHMARKS = {500, 2000, 5000}
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.8
EPSILON = 0.6
EPSILON_MIN = 0.1
NUM_EPISODES = 5000


def train_agent(num_episodes=5000):
    """
    Train the agent using Q-learning algorithm.
    """
    (
        average_steps_to_goal,
        env,
        episodes_indexes_lst,
        epsilon,
        epsilon_decay_rate,
        q_value_lookup_table,
        rewards_per_episode,
        state_s,
        steps_to_goal_per_episode
    ) = init_training(num_episodes)

    for episode_idx in range(1, num_episodes + 1):
        reward_per_episode = 0
        num_steps_to_goal = 0

        terminated = False
        truncated = False

        (num_steps_to_goal,
         reward_per_episode) = do_episode(env, epsilon, num_steps_to_goal, q_value_lookup_table,
                                          reward_per_episode, state_s, terminated, truncated)

        steps_to_goal_per_episode.append(num_steps_to_goal)

        if episode_idx % 100 == 0:
            prepare_average_lst(average_steps_to_goal, episode_idx, episodes_indexes_lst,
                                steps_to_goal_per_episode)

        epsilon = max(epsilon - epsilon_decay_rate, EPSILON_MIN)
        if episode_idx in EPISODES_BENCHMARKS:
            plot_heat_map(episode_idx, q_value_lookup_table)
        rewards_per_episode.append(reward_per_episode)
    do_plots(average_steps_to_goal, episodes_indexes_lst, rewards_per_episode)

    env.close()


def init_training(num_episodes):
    """
    This function initializes the parameters and environment for training the agent using the Q-learning algorithm.
    It sets up the FrozenLake environment with a 4x4 grid and no slipperiness.
    It initializes the Q-value lookup table, epsilon (exploration rate), epsilon decay rate,
    lists for tracking rewards and steps to goal per episode, and lists for tracking average steps to goal and episode indices.
    """
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)  # Init a MDP environments
    state_s = env.reset()[0]  # Resets the environment to an initial state,
    q_value_lookup_table = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = EPSILON
    epsilon_decay_rate = EPSILON / num_episodes
    rewards_per_episode = []  # Track rewards per episode
    steps_to_goal_per_episode = []  # Track number of steps to goal per episode
    average_steps_to_goal = []
    episodes_indxs_lst = []
    return average_steps_to_goal, env, episodes_indxs_lst, epsilon, epsilon_decay_rate, q_value_lookup_table, \
           rewards_per_episode, state_s, steps_to_goal_per_episode


def do_episode(env, epsilon, num_steps_to_goal, q_value_lookup_table, reward_per_episode, state_s, terminated,
               truncated):
    """
    This function performs a single episode of the environment by taking actions based on the epsilon-greedy policy.
    It updates the Q-values using the Q-learning update rule.
    If the episode is terminated or truncated, it resets the environment.
    @param env: (gym.Env): The environment.
    @param epsilon: (float) The exploration rate.
    @param num_steps_to_goal: (int) The number of steps taken to reach the goal.
    @param q_value_lookup_table: (numpy.ndarray) A 2D array representing the Q-values of each state-action pair.
    @param reward_per_episode: (float) The total reward obtained in the episode.
    @param state_s: (int) The current state.
    @param terminated: (bool) Flag indicating whether the episode terminated.
    @param truncated: (bool) Flag indicating whether the episode was truncated.
    @return: tuple: A tuple containing the number of steps taken to reach the goal and the total reward obtained.
    """
    while not terminated and not truncated:
        num_steps_to_goal += 1

        action = sample_action(env, epsilon, q_value_lookup_table, state_s)

        state_s_prime, reward, terminated, truncated, info = env.step(action)
        reward_per_episode += reward

        if terminated or truncated:
            target = reward
            state_s_prime = env.reset()[0]
        else:
            target = reward + DISCOUNT_FACTOR * np.max(q_value_lookup_table[state_s_prime, :])
        q_value_lookup_table[state_s][action] = (1 - LEARNING_RATE) * q_value_lookup_table[state_s][
            action] + LEARNING_RATE * target
        state_s = state_s_prime
    if truncated:  # The agent didn't get to the goal
        num_steps_to_goal = 100
    return num_steps_to_goal, reward_per_episode


def do_plots(average_steps_to_goal, episodes_indxs_lst, rewards_per_episode):
    plot_rewards_per_episode(rewards_per_episode)
    plot_avg_steps_per_100_episodes(episodes_indxs_lst, average_steps_to_goal)


def prepare_average_lst(average_steps_to_goal, episode_idx, episodes_indxs_lst, steps_to_goal_per_episode):
    """
    This function prepares lists for storing the average steps to goal per episode.
    It calculates the average steps to goal for the current episode and appends it to the list.
    It also appends the current episode index to the list of episode indices.
    Finally, it resets the list for storing steps to goal per episode for the next episode.
    """
    avg_steps = np.mean(steps_to_goal_per_episode)
    average_steps_to_goal.append(avg_steps)
    episodes_indxs_lst.append(episode_idx)
    steps_to_goal_per_episode = []
    return steps_to_goal_per_episode


def sample_action(env, epsilon, lookup_table, state_s):
    """
     This function selects an action using an epsilon-greedy policy:
      - With probability epsilon, it selects a random action (exploration).
      - With probability (1 - epsilon), it selects the action with the highest Q-value for the given state (exploitation).
    @return: action (int): The selected action.

    """
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # agent policy that uses the observation and info
    else:
        action = np.argmax(lookup_table[state_s, :])
    return action


def plot_heat_map(episode_idx, lookup_table):
    """
    This function creates a heat map representing the Q-values of each state-action pair.
    The x-axis represents the action indices (left, down, right, up),
    and the y-axis represents the state indices.
    """
    plt.imshow(lookup_table, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar to show the mapping of values to colors

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(lookup_table.shape[1]), np.arange(lookup_table.shape[0]))

    # Annotate the plot with each value in the array
    for x, y, value in zip(x_coords.flatten(), y_coords.flatten(), lookup_table.flatten()):
        plt.text(x, y, "{:.2f}".format(value), ha='center', va='center', color='white', fontsize=6)
        # Set custom x-axis tick labels
    plt.xticks(np.arange(lookup_table.shape[1]), ['left', 'down', 'right', 'up'], fontsize=6)
    plt.text(lookup_table.shape[1] / 2, -2, f"number of episodes = {episode_idx}",
             ha="center", va="center", fontsize=8)
    plt.savefig(f'heat_map_{episode_idx}.jpg', format='jpeg')  # Adjust quality as needed
    plt.show()


def plot_rewards_per_episode(rewards_per_episode):
    """
    This function plots the total reward achieved in each episode.
    The x-axis represents the episode number, and the y-axis represents the total reward.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.grid()
    plt.savefig('rewards_per_episode.jpg', format='jpeg')
    plt.show()


def plot_avg_steps_per_100_episodes(episode_idx, avg_steps):
    """
    This function plots the average steps to goal over the last 100 episodes
    @param episode_idx: List of episode indices
    @param avg_steps: List of average steps to goal for each episode.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(episode_idx, avg_steps, 'ro')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps to Goal (Last 100 Episodes)')
    plt.title('Average Steps to Goal over Last 100 Episodes')
    plt.grid()
    plt.savefig('avg_steps_per_100_episodes.jpg', format='jpeg')
    plt.show()


if __name__ == '__main__':
    train_agent(NUM_EPISODES)
