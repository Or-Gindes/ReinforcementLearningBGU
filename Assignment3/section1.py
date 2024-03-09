import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
from actor_critic import ActorCritic
import tensorboard

BASE_ENV = 'CartPole-v1'
TRANSFER_ENVS = ['Acrobot-v1', 'MountainCarContinuous-v0']
ENV_NAMES = [BASE_ENV] + TRANSFER_ENVS

MAX_ENV_STEPS = {
    'CartPole-v1': 500,
    'Acrobot-v1': 500,
    'MountainCarContinuous-v0': 999
}

# Define training hyperparameters
MAX_EPISODES = 5000
DISCOUNT_FACTOR = 1.0
LEARNING_RATE = 4e-4


def pad_state(state, pad2len):
    if state.shape[0] < pad2len:
        state = np.pad(state, (0, pad2len - state.shape[0]), mode='constant')

    return state.reshape([1, pad2len])


def mask_actions(actions_distribution, num_actions):
    action_mask = np.ones_like(actions_distribution)
    action_mask[num_actions:] = 0
    masked_actions_dist = actions_distribution * action_mask
    masked_actions_dist /= np.sum(masked_actions_dist)
    return masked_actions_dist


def run():
    environments = {env_name: gym.make(env_name) for env_name in ENV_NAMES}
    env_params = {
        env_name: {
            "state_size": environments[env_name].observation_space.shape[0],
            "action_size": environments[env_name].action_space.n if env_name != 'MountainCarContinuous-v0' else 3
        } for env_name in ENV_NAMES
    }

    # state_size = input_size
    standardized_state_size = max([params['state_size'] for params in env_params.values()])
    # action_size = output_size
    standardized_action_size = max([params['action_size'] for params in env_params.values()])

    train_base_env(environments[BASE_ENV], standardized_state_size, standardized_action_size)
    # transfer_learning(env1)
    # transfer_learning(env2)


def train_base_env(env, state_size, action_size):
    # Initialize the policy network
    tf.reset_default_graph()
    policy = ActorCritic(state_size, action_size, LEARNING_RATE, DISCOUNT_FACTOR)

    # Start training the ActorCritic network
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(f"./logs/{BASE_ENV}", sess.graph)
        solved = False
        episode_rewards = np.zeros(MAX_EPISODES)
        average_rewards = 0.0
        max_average_rewards = 0.0
        patience = 50
        patience_counter = 0

        for episode in range(MAX_EPISODES):
            episode_loss = 0
            state, _ = env.reset()
            state = pad_state(state, state_size)
            I = 1.0
            done = False

            while not done:
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                masked_actions_dist = mask_actions(actions_distribution, env.action_space.n)
                action = np.random.choice(np.arange(len(masked_actions_dist)), p=masked_actions_dist)

                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = pad_state(next_state, state_size)
                done = terminated or truncated

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1

                feed_dict = {policy.state: state, policy.R_t: reward, policy.action: action_one_hot,
                             policy.nnext_state: next_state, policy.done: done, policy.I: I}
                _, _, loss = sess.run([policy.optimizer, policy.value_network_optimizer, policy.loss], feed_dict)
                episode_loss += loss
                I = DISCOUNT_FACTOR * I

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards < max_average_rewards:
                        if patience_counter > patience:
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        else:
                            patience_counter += 1
                    else:
                        max_average_rewards = average_rewards

                state = next_state

            # if problem is solved, i.e. training converged, break out of the episode training loop
            if solved:
                break

            policy.write_summary(sess, writer, average_rewards, episode_rewards[episode], episode_loss, episode)

        writer.close()


if __name__ == '__main__':
    run()
