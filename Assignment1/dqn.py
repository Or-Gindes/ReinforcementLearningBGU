"""
This script is an implementation of an agent using the basic DQN algorithm
Reinforcement Learning course - Assignment 1 - Section 2
"""
import os
from typing import Optional, Tuple, List, Union, SupportsFloat
import gymnasium
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from collections import deque
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

ENVIRONMENT = "CartPole-v1"
LEARNING_RATE = 0.001
DQN_HIDDEN_3 = (512, 256, 64)
DQN_HIDDEN_5 = (512, 256, 128, 64, 32)
REPLAY_MEMORY_SIZE = 1e6

M_EPISODES = 7500
C_STEPS_UPDATE = 50
DISCOUNT_FACTOR = 0.995
BATCH_SIZE = 32
EPSILON = 1
EPSILON_DECAY = 0.999
EPSILON_MIN = 0


class DQN:
    """This class implements the DQN neural network"""

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            lr: float = LEARNING_RATE,
            optimizer=Adam,
            loss=MSE,
            layers: Tuple[int, ...] = DQN_HIDDEN_3,
    ) -> None:
        """
        :param num_states: number of states possible in the estimated environment
        :param num_actions: number of actions possible in the estimated environment
        :param lr: learning rate
        :param optimizer: tensorflow optimizer object
        :param loss: tensorflow loss object or a valid name of loss in string
        :param layers: Tuple of ints representing layer sizes to build the model with
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self._build_model(layers, optimizer, loss)

    def _build_model(self, layers: Tuple[int, ...], optimizer, loss) -> None:
        """
        Method to build the DQN model according to layers input
        The model is the approximation of the action-value function Q and is initialized by default with random weights
        :param layers: Tuple of layer sizes to build the model with
        :param optimizer: tensorflow optimizer object or a valid name of optimizer in string
        :param loss: tensorflow loss object or a valid name of loss in string
        """
        model = Sequential()
        input_layer = Dense(layers[0], input_dim=self.num_states, activation="relu")
        model.add(input_layer)

        for layer_size in layers[1:]:
            model.add(Dense(units=layer_size, activation="relu"))

        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(loss=loss, optimizer=optimizer(learning_rate=self.lr))
        self.model = model

    def sample_action(self, state: np.array, epsilon: float) -> int:
        """
        with probability epsilon select random action otherwise select greedily according to q_values
        :param state: an array of floats describing the current environment state
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :return: the number of the action that should be taken according to decaying ε-greedy
        """
        if np.random.rand() <= epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.predict(state.reshape(1, -1))
        action = tf.argmax(q_values, axis=1).numpy()[0]
        return action

    def predict(self, state: tf.Tensor) -> tf.Tensor:
        """Predict q-values using the Q_function
        :param state: an array of floats describing the current environment state
        :return: q_values predicted from model which estimates the Q function
        """
        q_values = self.model.predict(state, verbose=0)
        return q_values


class ExperienceReplay:
    """Implementation of a memory buffer - a deque in a fixed size to store past experiences"""

    def __init__(self, size: int) -> None:
        """
        :param size: size of the deque (int)
        """
        self._memory_buffer = deque(maxlen=int(size))

    def add(
            self,
            state: np.array,
            action: int,
            reward: SupportsFloat,
            next_state: np.array,
            terminated: bool,
    ) -> None:
        """
        :param state: current state
        :param action: current action
        :param reward: reward for current (state, action) pair
        :param next_state: state which the current (state, action) pair lead to
        :param terminated: whether the simulation was completed by the action
        """
        self._memory_buffer.append((state, action, reward, next_state, terminated))

    def sample_batch(
            self, batch_size: int
    ) -> Optional[Tuple[Union[np.array, int, float, bool]]]:
        """
        :param batch_size: int - size of batch to sample from the memory buffer
        :return: if there are enough samples to return (i.e. more than batch_size) sample randomly and return
        """
        if len(self._memory_buffer) < batch_size:
            return

        sample = random.sample(self._memory_buffer, batch_size)
        return sample


class DeepQLearning:
    """This class is in charge of executing the DQN algorithm"""

    def __init__(self, env=ENVIRONMENT, dqn_model=DQN, dqn_layers=DQN_HIDDEN_3, lr=LEARNING_RATE, optim=Adam, loss=MSE):
        """
        :param env: the environment which the networks are learning
        :param dqn_model: The model used in the algorithm
        :param dqn_layers: how to build the model
        :param lr: learning rate for the model optimizer
        :param optim: compatible optimizer class
        :param loss: compatible loss class
        """
        self.setup_gpu()
        # Load environment and check it state and action parameters
        self.env = gymnasium.make(env)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.dqn = dqn_model
        self.dqn_layers = dqn_layers
        self.lr = lr
        self.optim = optim
        self.loss = loss

        self._build_dqn()

    def _build_dqn(self):
        """Build Agent and Target deep q learning networks"""
        # Build two DQN networks, the agent and the "fixed" target
        self.agent = self.dqn(self.n_states, self.n_actions, lr=self.lr, optimizer=self.optim, loss=self.loss,
                              layers=self.dqn_layers)
        self.target = self.dqn(self.n_states, self.n_actions, lr=self.lr, optimizer=self.optim, loss=self.loss,
                               layers=self.dqn_layers)
        # Verify the Q-function is initialized the same in both agent and target
        self.target.model.set_weights(self.agent.model.get_weights())

    def train_agent(
            self,
            gamma: float = DISCOUNT_FACTOR,
            replay_size: int = REPLAY_MEMORY_SIZE,
            batch_size: int = BATCH_SIZE,
            epsilon: float = EPSILON,
            epsilon_decay: float = EPSILON_DECAY,
    ) -> Tuple[List[float], List[int]]:
        """
        :param gamma: discount factor
        :param replay_size: size of the experience replay buffer
        :param batch_size: size of batches to be sampled form the experience replay buffer
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :param epsilon_decay: rate of epsilon decay
        :return: a tuple of lists. The first, a list of average loss value per episode.
        The second, a list of rewards accumulated per episode
        """
        experience_replay = ExperienceReplay(replay_size)
        avg_episode_losses, episode_rewards = [], []
        # first_threshold_episode is the first episode where the agent obtains criteria
        first_threshold_episode = None

        tqdm_episodes = tqdm(range(0, M_EPISODES), desc="Episode:")
        for episode in tqdm_episodes:
            state, _ = self.env.reset()
            terminated = False
            episode_reward = 0
            epsilon = max(epsilon * epsilon_decay, EPSILON_MIN)
            loss = []
            steps_since_update = 0

            while not terminated:
                # select action with prob epsilon of being random
                action = self.agent.sample_action(state, epsilon)
                # execute action in the environment and observe new state and reward
                next_state, reward, terminated, _, _ = self.env.step(action)
                episode_reward += reward

                # Store the transition in replay memory
                experience_replay.add(state, action, reward, next_state, terminated)

                # After the transition is stored, update current state to the next state S_t+1 = S_t
                state = next_state

                # sample a minibatch of transitions from replay memory
                transitions_minibatch = experience_replay.sample_batch(batch_size)
                if not transitions_minibatch:
                    continue

                # Unpack the minibatch into separate tensors
                minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_terminated = (
                    zip(*transitions_minibatch))

                minibatch_states = tf.convert_to_tensor(minibatch_states, dtype=tf.float32)
                minibatch_actions = tf.convert_to_tensor(minibatch_actions, dtype=tf.int32)
                minibatch_rewards = tf.convert_to_tensor(minibatch_rewards, dtype=tf.float32)
                minibatch_next_states = tf.convert_to_tensor(minibatch_next_states, dtype=tf.float32)
                minibatch_terminated = tf.convert_to_tensor(minibatch_terminated, dtype=tf.bool)

                target_q_values = minibatch_rewards + gamma * tf.reduce_max(self.target.predict(minibatch_next_states), axis=1)
                minibatch_y = tf.where(minibatch_terminated, minibatch_rewards, target_q_values)

                # perform gradient decent step on MSE loss (model compiled with MSE)
                history = self.agent.model.fit(
                    x=minibatch_states, y=minibatch_y, batch_size=batch_size, verbose=0,
                    use_multiprocessing=True, workers=os.cpu_count()
                )
                loss.append(history.history["loss"][0])

                # if steps reached number of steps for update
                steps_since_update += 1
                if steps_since_update == C_STEPS_UPDATE:
                    steps_since_update = 0
                    self.target.model.set_weights(self.agent.model.get_weights())

            if loss:
                avg_episode_loss = sum(loss) / len(loss)
                tqdm_episodes.set_postfix(average_episode_loss=avg_episode_loss, episode_reward=episode_reward)
                # print(f"Episode: {episode + 1} | Loss: {avg_episode_loss:.4f} | Reward: {episode_reward:.2f}")
                # when the episode is finished, log the average loss of the episode
                avg_episode_losses.append(avg_episode_loss)
                episode_rewards.append(episode_reward)

            # Number of episodes until the agent obtains an average reward >= 475 over 100 consecutive episodes
            if not first_threshold_episode and sum(episode_rewards[-100:]) / 100 >= 475.0:
                first_threshold_episode = episode + 1

        if first_threshold_episode:
            print(f"Number of episodes to achieve average reward "
                  f"of at least 475 over 100 consecutive episodes: {first_threshold_episode}")

        return avg_episode_losses, episode_rewards

    def test_agent(self, render: bool = True):
        """
        Test the trained agent in env on a new episode and render the environment
        :param render: boolean parameter indicating whether to render the agent playing the environment
        """
        state, _ = self.env.reset()
        # The model is trained and doesn't require any more exploration when selecting actions in the environment
        epsilon = 0
        terminated = False
        total_reward = 0
        while not terminated:
            # select action
            action = self.agent.sample_action(state, epsilon)
            # execute action in the environment and observe new state and reward
            next_state, reward, terminated, _, _ = self.env.step(action)
            total_reward += reward
            if render:
                self.env.render()

            state = next_state

        print(f"Score of the trained agent in a new episode = {total_reward}")

    @staticmethod
    def setup_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU setup success")
            except RuntimeError as e:
                print("Failed to setup GPU")


# TODO: Play around with params and introduce a loop where we only plot at the end.
#  plot all loss / rewards together and add legend with hyperparameters
def plot_training_graphs(losses: List[float], rewards: List[int], results_dir: str = "DQN_graphs"):
    """
    Function plots the average losses and rewards recorded per training episode
    :param losses: list of average losses per training episode
    :param rewards: list of total rewards scored per training episode
    :param results_dir: folder to save plots in
    """
    full_path = os.path.join(os.getcwd(), results_dir)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    # Plot losses -
    plt.plot(x=range(len(losses)), y=losses)
    plt.title("DQN - Average MSE Loss per Training Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average MSE Loss")
    plt.savefig(os.path.join(full_path, "Losses_per_Episode"))

    plt.clf()
    # Plot rewards -
    plt.plot(x=range(len(rewards)), y=rewards)
    plt.title("DQN - Total Reward per Training Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(full_path, "Rewards_per_Episode.png"))


def main():
    # TODO: Read training configuration and generate multiple graphs
    dqn = DeepQLearning()
    avg_episode_losses, episode_rewards = dqn.train_agent()
    plot_training_graphs(avg_episode_losses, episode_rewards)
    dqn.test_agent()


if __name__ == "__main__":
    main()
