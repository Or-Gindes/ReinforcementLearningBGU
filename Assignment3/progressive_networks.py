import tensorflow.compat.v1 as tf
from individual_networks import *
from actor_critic import ActorCritic
import os

import gymnasium as gym
import numpy as np

##from here my try

# with tf.Session() as sess:
#
#     saver = tf.train.import_meta_graph(os.path.join(BASE_MODEL_PATH, "Acrobot-v1_model.ckpt.meta"))
#
#     source_model_path = os.path.join(BASE_MODEL_PATH, 'Acrobot-v1_model.ckpt')
#     saver.restore(sess, source_model_path)
#     sess.run(tf.assign(policy.W1, tf.stop_gradient(policy.W1)))


# until here my try


# new_saver = tf.train.import_meta_graph(os.path.join(BASE_MODEL_PATH, "Acrobot-v1_model.ckpt.meta"))
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))

# # Initialize variables
# sess.run(tf.global_variables_initializer())
#
# # Restore the model
# #saver = tf.train.Saver()
# saver.restore(sess, os.path.join(BASE_MODEL_PATH, "Acrobot-v1_model.ckpt"))  # Provide the path to the checkpoint file

# sess=tf.Session()
#
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph(os.path.join(BASE_MODEL_PATH, "Acrobot-v1_model.ckpt"))
# saver.restore(sess,tf.train.latest_checkpoint('./'))


# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

np.random.seed(1)

tf.compat.v1.set_random_seed(1)

PROGRESSIVE_SCENARIO1 = {"source1": ACROBOT, "source2": MOUNTAINCAR, "target": CARTPOLE}
PROGRESSIVE_SCENARIO2 = {"source1": CARTPOLE, "source2": ACROBOT, "target": MOUNTAINCAR}

PROGRESSIVE_SCENARIOS = [PROGRESSIVE_SCENARIO1, PROGRESSIVE_SCENARIO2]


def run():
    for scenario in PROGRESSIVE_SCENARIOS:
        source1_env_name, source2_env_name, target_env_name = (scenario["source1"],
                                                               scenario["source2"],
                                                               scenario["target"])
        train_prog_env(source1_env_name, source2_env_name, target_env_name)
        break


def train_prog_env(source1_env_name, source2_env_name, target_env_name):
    target_env = gym.make(target_env_name)
    tf.reset_default_graph()

    converge_thresh = CONVERGENCE_THRESHOLD[target_env_name]

    # Start training the ActorCritic network
    with tf.Session() as sess:
        with tf.compat.v1.variable_scope(f"{source1_env_name}_model", reuse=False):
            source1_policy = ActorCritic(STANDARDIZED_STATE_SIZE, STANDARDIZED_ACTION_SIZE, 5e-3, 1)
            saver1 = tf.train.Saver(var_list=source1_policy.variables + source1_policy.critic.variables)
            source1_model_path = os.path.join(BASE_MODEL_PATH, f'{source1_env_name}' + "_model.ckpt")

            saver1.restore(sess, source1_model_path)

            # freeze the gradients of the initial layers in the actor and the critic networks
            sess.run(tf.assign(source1_policy.W1, tf.stop_gradient(source1_policy.W1)))
            sess.run(tf.assign(source1_policy.b1, tf.stop_gradient(source1_policy.b1)))
            sess.run(tf.assign(source1_policy.critic.W1, tf.stop_gradient(source1_policy.critic.W1)))
            sess.run(tf.assign(source1_policy.critic.b1, tf.stop_gradient(source1_policy.critic.b1)))

            sess.run(tf.assign(source1_policy.W2, tf.stop_gradient(source1_policy.W2)))
            sess.run(tf.assign(source1_policy.b2, tf.stop_gradient(source1_policy.b2)))
            sess.run(tf.assign(source1_policy.critic.W2, tf.stop_gradient(source1_policy.critic.W2)))
            sess.run(tf.assign(source1_policy.critic.b2, tf.stop_gradient(source1_policy.critic.b2)))

        with tf.compat.v1.variable_scope(f"{source2_env_name}_model", reuse=False):
            source2_policy = ActorCritic(STANDARDIZED_STATE_SIZE, STANDARDIZED_ACTION_SIZE, 5e-3, 1)
            saver2 = tf.train.Saver(var_list=source2_policy.variables + source2_policy.critic.variables)

            source2_model_path = os.path.join(BASE_MODEL_PATH, f'{source2_env_name}' + "_model.ckpt")

            saver2.restore(sess, source2_model_path)

            ############### ~~~~~~~~~~~~~~~~~~~ ###############

            sess.run(tf.assign(source2_policy.W1, tf.stop_gradient(source2_policy.W1)))
            sess.run(tf.assign(source2_policy.b1, tf.stop_gradient(source2_policy.b1)))
            sess.run(tf.assign(source2_policy.critic.W1, tf.stop_gradient(source2_policy.critic.W1)))
            sess.run(tf.assign(source2_policy.critic.b1, tf.stop_gradient(source2_policy.critic.b1)))

            sess.run(tf.assign(source2_policy.W2, tf.stop_gradient(source2_policy.W2)))
            sess.run(tf.assign(source2_policy.b2, tf.stop_gradient(source2_policy.b2)))
            sess.run(tf.assign(source2_policy.critic.W2, tf.stop_gradient(source2_policy.critic.W2)))
            sess.run(tf.assign(source2_policy.critic.b2, tf.stop_gradient(source2_policy.critic.b2)))

        # # re-initialize the output layer weights for the networks
        # tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
        # sess.run(policy.W2.assign(tf2_initializer(policy.W2.shape)))
        # sess.run(policy.b2.assign(tf.zeros_initializer()(policy.b2.shape)))
        # sess.run(policy.critic.W2.assign(tf2_initializer(policy.critic.W2.shape)))
        # sess.run(policy.critic.b2.assign(tf.zeros_initializer()(policy.critic.b2.shape)))

        policy = progressiveActorCritic(STANDARDIZED_STATE_SIZE, STANDARDIZED_ACTION_SIZE, 5e-3,
                                        source1_policy, source2_policy, 1)

        # writer = tf.summary.FileWriter(f"./logs/section2/{source1_env_name}->{target_env_name}", sess.graph)
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

            # policy.write_summary(sess, writer, average_rewards, episode_rewards[episode], episode_loss, episode)

        # writer.close()


def setup_summary():
    episode_avg_reward = tf.Variable(0.)
    episode_loss = tf.Variable(0.)
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("average reward over 100 episodes", episode_avg_reward)
    tf.summary.scalar("episode reward", episode_reward)
    tf.summary.scalar("episode loss", episode_loss)
    summary_vars = [episode_avg_reward, episode_reward, episode_loss]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op


class StateValuesNetwork:
    def __init__(self, state_size, learning_rate, state, next_state, done, name='critic'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.state = state
        self.next_state = next_state
        self.done = done

        with tf.variable_scope(name):
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.state_output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.Z2 = tf.add(tf.matmul(self.next_state, self.W1), self.b1)
            self.A2 = tf.nn.relu(self.Z2)

            self.next_state_output = tf.multiply(
                tf.add(tf.matmul(self.A2, self.W2), self.b2),
                tf.subtract(tf.constant(1.0), self.done)
            )

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


class progressiveActorCritic:
    def __init__(self, state_size, action_size, learning_rate, source1_model, source2_model, discount_factor=0.99,
                 name='actor_critic'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.summary_placeholders, self.update_ops, self.summary_op = setup_summary()

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.nnext_state = tf.placeholder(tf.float32, [None, self.state_size], name="nnext_state")
            self.done = tf.placeholder(tf.float32, name='done')
            self.I = tf.placeholder(tf.float32, name="I")

            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 32], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [32], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [32, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            concatenated_hidden = tf.concat([source1_model.A1, source2_model.A1], axis=1)

            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.output = tf.concat([concatenated_hidden, self.output], axis=1)
            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))

            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)

            self.critic = StateValuesNetwork(self.state_size, self.learning_rate * 10, self.state, self.nnext_state,
                                             self.done)
            self.delta = self.R_t + (self.discount_factor * self.critic.next_state_output) - self.critic.state_output
            self.loss = tf.reduce_mean(self.neg_log_prob * self.I * tf.stop_gradient(self.delta))
            self.value_network_optimizer = self.critic.optimizer.minimize(self.delta ** 2)

            # Loss with negative log probability
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def write_summary(self, sess, writer, avg_episode_reward, episode_reward, episode_loss, episode):
        summary_values = {self.summary_placeholders[0]: avg_episode_reward,
                          self.summary_placeholders[1]: episode_reward, self.summary_placeholders[2]: episode_loss}
        sess.run(self.update_ops, feed_dict=summary_values)
        summary_str = sess.run(self.summary_op, feed_dict=summary_values)
        writer.add_summary(summary_str, episode)
        writer.flush()


if __name__ == '__main__':
    run()
