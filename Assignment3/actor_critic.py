import os

import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

np.random.seed(1)

ALL_ENV = ['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0']


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
    def __init__(self, state_size, learning_rate, state, next_state=None, done=None, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.state = state
        self.next_state = next_state
        self.done = done

        with tf.variable_scope(name):
            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 32], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [32], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [32, 1], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.state_output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            if not (self.next_state is None):
                self.Z2 = tf.add(tf.matmul(self.next_state, self.W1), self.b1)
                self.A2 = tf.nn.relu(self.Z2)

                self.next_state_output = tf.multiply(
                    tf.add(tf.matmul(self.A2, self.W2), self.b2),
                    tf.subtract(tf.constant(1.0), self.done)
                )

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, discount_factor=0.99, name='policy_network',
                 baseline=True, critic=False):
        assert not baseline or not critic, "REINFORCE can be run with either baseline OR a critic, not both"
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
            self.W1 = tf.get_variable("W1", [self.state_size, 16], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [16], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [16, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))

            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)

            self.critic = StateValuesNetwork(self.state_size, self.learning_rate / 2, self.state, self.nnext_state,
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


def run():
    train_base_env()
    transfer_learning(env1)
    transfer_learning(env2)


def train_base_env():
    # Define environment
    render = False
    env_name = "CartPole-v1"
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define training hyperparameters
    max_episodes = 5000
    max_steps = 501
    discount_factor = 1.0
    learning_rate = 4e-4

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate, discount_factor=discount_factor)

    # Start training the agent with REINFORCE w/baseline algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(f"./logs/{env_name}", sess.graph)
        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        max_average_rewards = 0.0
        patience = 50
        patience_counter = 0

        for episode in range(max_episodes):
            episode_loss = 0
            state, _ = env.reset()
            state = state.reshape([1, state_size])
            I = 1.0

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])
                done = terminated or truncated

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1

                feed_dict = {policy.state: state, policy.R_t: reward, policy.action: action_one_hot,
                             policy.nnext_state: next_state, policy.done: done, policy.I: I}
                _, _, loss = sess.run([policy.optimizer, policy.value_network_optimizer, policy.loss], feed_dict)
                episode_loss += loss
                I = discount_factor * I

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
                    break

                state = next_state

            if solved:
                break

            policy.write_summary(sess, writer, average_rewards, episode_rewards[episode], episode_loss, episode)

        writer.close()


if __name__ == '__main__':
    run()
