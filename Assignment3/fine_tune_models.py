import os
import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
from individual_networks import *
from actor_critic import ActorCritic

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

TRANSFER_SCENARIO1 = {"source": ACROBOT, "target": CARTPOLE}
TRANSFER_SCENARIO2 = {"source": CARTPOLE, "target": MOUNTAINCAR}

FINE_TUNE_SCENARIOS = [TRANSFER_SCENARIO1 , TRANSFER_SCENARIO2]


def run():
    for scenario in FINE_TUNE_SCENARIOS:
        source_env_name, target_env_name = scenario["source"], scenario["target"]
        fine_tune(source_env_name, target_env_name)


def fine_tune(source_env_name, target_env_name):
    target_env = gym.make(target_env_name)
    tf.reset_default_graph()
    policy = ActorCritic(STANDARDIZED_STATE_SIZE, STANDARDIZED_ACTION_SIZE, 5e-3, 1)

    converge_thresh = CONVERGENCE_THRESHOLD[target_env_name]
    saver = tf.train.Saver(var_list=tf.global_variables())
    # Start training the ActorCritic network
    with tf.Session() as sess:
        source_model_path = os.path.join(BASE_MODEL_PATH, f'{source_env_name}' + "_model.ckpt")

        saver.restore(sess, source_model_path)

        # freeze the gradients of the initial layers in the actor and the critic networks
        sess.run(tf.assign(policy.W1, tf.stop_gradient(policy.W1)))
        sess.run(tf.assign(policy.b1, tf.stop_gradient(policy.b1)))
        sess.run(tf.assign(policy.critic.W1, tf.stop_gradient(policy.critic.W1)))
        sess.run(tf.assign(policy.critic.b1, tf.stop_gradient(policy.critic.b1)))

        # re-initialize the output layer weights for the networks
        tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
        sess.run(policy.W2.assign(tf2_initializer(policy.W2.shape)))
        sess.run(policy.b2.assign(tf.zeros_initializer()(policy.b2.shape)))
        sess.run(policy.critic.W2.assign(tf2_initializer(policy.critic.W2.shape)))
        sess.run(policy.critic.b2.assign(tf.zeros_initializer()(policy.critic.b2.shape)))

        writer = tf.summary.FileWriter(f"./logs/section2/{source_env_name}->{target_env_name}", sess.graph)
        solved = False
        episode_rewards = np.zeros(MAX_EPISODES)
        average_rewards = -1e3

        if target_env_name == MOUNTAINCAR:
            num_goal_reached = 0
            scaler = scale_observations(target_env, STANDARDIZED_STATE_SIZE)

        for episode in range(MAX_EPISODES):
            episode_loss = 0
            state, _ = target_env.reset()
            state = pad_state(state, STANDARDIZED_STATE_SIZE)
            I = 1.0
            done = False

            if target_env_name == MOUNTAINCAR:
                state = scaler.transform(state)
                max_left = max_right = state[0, 0]

            while not done:
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                masked_actions_dist = mask_actions(actions_distribution, ENV_PARAMS[target_env_name]['action_size'])
                action = np.random.choice(np.arange(len(masked_actions_dist)), p=masked_actions_dist)

                if target_env_name == MOUNTAINCAR:
                    continuous_action = [MOUNTAINCAR_DISCRETE_TO_CONTINUOUS[action]]
                    next_state, reward, terminated, truncated, _ = target_env.step(continuous_action)
                    next_state = scaler.transform(pad_state(next_state, STANDARDIZED_STATE_SIZE))
                else:
                    next_state, reward, terminated, truncated, _ = target_env.step(action)
                    next_state = pad_state(next_state, STANDARDIZED_STATE_SIZE)

                done = terminated or truncated

                action_one_hot = np.zeros(STANDARDIZED_ACTION_SIZE)
                action_one_hot[action] = 1

                episode_rewards[episode] += reward

                # Auxiliary rewards per height achieved by the car
                if target_env_name == MOUNTAINCAR:
                    if num_goal_reached < 20:
                        if reward <= 0:
                            if next_state[0, 0] < max_left:
                                reward = (2 + next_state[0, 0]) ** 2
                                max_left = next_state[0, 0]

                            if next_state[0, 0] > max_right:
                                reward = (2 + next_state[0, 0]) ** 2
                                max_right = next_state[0, 0]

                        else:
                            num_goal_reached += 1
                            reward += 100
                            print(f'goal reached {num_goal_reached} times')

                if target_env_name == CARTPOLE:
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


if __name__ == '__main__':
    run()
