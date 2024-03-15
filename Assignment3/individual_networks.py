import os
import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
from actor_critic import ActorCritic
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

CARTPOLE = 'CartPole-v1'
ACROBOT = 'Acrobot-v1'
MOUNTAINCAR = 'MountainCarContinuous-v0'
ENV_NAMES = [CARTPOLE, ACROBOT, MOUNTAINCAR]

ENVIRONMENTS = {env_name: gym.make(env_name) for env_name in ENV_NAMES}
ENV_PARAMS = {
    env_name: {
        "state_size": ENVIRONMENTS[env_name].observation_space.shape[0],
        "action_size": ENVIRONMENTS[env_name].action_space.n if env_name != MOUNTAINCAR else 3
    } for env_name in ENV_NAMES
}

# state_size = input_size
STANDARDIZED_STATE_SIZE = max([params['state_size'] for params in ENV_PARAMS.values()])
# action_size = output_size
STANDARDIZED_ACTION_SIZE = max([params['action_size'] for params in ENV_PARAMS.values()])

MAX_ENV_STEPS = {
    CARTPOLE: 500,
    ACROBOT: 500,
    MOUNTAINCAR: 999
}

CONVERGENCE_THRESHOLD = {
    CARTPOLE: 475,
    ACROBOT: -90,
    MOUNTAINCAR: 75
}

MOUNTAINCAR_DISCRETE_TO_CONTINUOUS = {0: -1.0, 1: 0.0, 2: 1.0}

# Define training hyperparameters
MAX_EPISODES = 5000
DISCOUNT_FACTOR = 1
LEARNING_RATE = 1e-3

BASE_MODEL_PATH = "./saved_models/"


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


def scale_observations(env, padding_size):
    observation_examples = np.array([pad_state(env.observation_space.sample(), padding_size) for _ in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples.reshape(-1, padding_size))
    return scaler


def run():
    for env_name in ENV_NAMES:
        train_env(
            ENVIRONMENTS[env_name],
            env_name,
            STANDARDIZED_STATE_SIZE,
            STANDARDIZED_ACTION_SIZE,
            ENV_PARAMS
        )


def train_env(env, env_name, state_size, action_size, env_params, model_path=None):
    # Initialize the policy network
    tf.reset_default_graph()
    policy = ActorCritic(state_size, action_size, LEARNING_RATE, DISCOUNT_FACTOR)

    converge_thresh = CONVERGENCE_THRESHOLD[env_name]
    saver = tf.train.Saver()
    # Start training the ActorCritic network
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(f"./logs/section1/{env_name}", sess.graph)
        solved = False
        episode_rewards = np.zeros(MAX_EPISODES)
        average_rewards = -1e3

        if env_name == MOUNTAINCAR:
            num_goal_reached = 0
            scaler = scale_observations(env, state_size)

        for episode in range(MAX_EPISODES):
            episode_loss = 0
            state, _ = env.reset()
            state = pad_state(state, state_size)
            I = 1.0
            done = False

            if env_name == MOUNTAINCAR:
                state = scaler.transform(state)
                max_left = max_right = state[0, 0]

            while not done:
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                masked_actions_dist = mask_actions(actions_distribution, env_params[env_name]['action_size'])
                action = np.random.choice(np.arange(len(masked_actions_dist)), p=masked_actions_dist)

                if env_name == MOUNTAINCAR:
                    continuous_action = [MOUNTAINCAR_DISCRETE_TO_CONTINUOUS[action]]
                    next_state, reward, terminated, truncated, _ = env.step(continuous_action)
                    next_state = scaler.transform(pad_state(next_state, state_size))
                else:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = pad_state(next_state, state_size)

                done = terminated or truncated

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1

                episode_rewards[episode] += reward

                # Auxiliary rewards per height achieved by the car
                if env_name == MOUNTAINCAR:
                    if num_goal_reached < 20:
                        if reward <= 0:
                            if next_state[0, 0] < max_left:
                                reward = (1 + next_state[0, 0]) ** 2
                                max_left = next_state[0, 0]

                            if next_state[0, 0] > max_right:
                                reward = (1 + next_state[0, 0]) ** 2
                                max_right = next_state[0, 0]

                        else:
                            num_goal_reached += 1
                            reward += 100
                            print(f'goal reached {num_goal_reached} times')

                if env_name == ACROBOT:
                    reward = (reward - 50) / MAX_ENV_STEPS[ACROBOT]

                if env_name == CARTPOLE:
                    reward = 1 - reward / MAX_ENV_STEPS[CARTPOLE]

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
                if average_rewards >= converge_thresh:
                    print(' Solved at episode: ' + str(episode))
                    solved = True

                state = next_state

            # if problem is solved, i.e. training converged, break out of the episode training loop
            if solved:
                break

            policy.write_summary(sess, writer, average_rewards, episode_rewards[episode], episode_loss, episode)

        writer.close()

        if model_path is None:
            saver.save(sess, os.path.join(BASE_MODEL_PATH, f'{env_name}' + "_model.ckpt"))


if __name__ == '__main__':
    run()
