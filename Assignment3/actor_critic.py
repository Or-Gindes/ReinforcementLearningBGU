import os

import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

np.random.seed(1)



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
            self.W1 = tf.get_variable("W1", [self.state_size, 16], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [16], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [16, 1], initializer=tf2_initializer)
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


class ActorCritic:
    def __init__(self, state_size, action_size, learning_rate, discount_factor=0.99, name='actor_critic'):
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

