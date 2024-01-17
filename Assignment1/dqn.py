"""
This script is an implementation of an agent using the basic DQN algorithm
Reinforcement Learning course - Assignment 1 - Section 2
"""

from typing import Optional, Tuple, List, Union
import gymnasium
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from collections import deque
import random
import matplotlib.pyplot as plt

ENVIRONMENT = "CartPole-v1"
LEARNING_RATE = 0.001
DQN_HIDDEN_3 = (256, 64, 32)
DQN_HIDDEN_5 = (512, 256, 128, 64, 32)
REPLAY_MEMORY_SIZE = 1e5

M_EPISODES = 1000
C_STEPS_UPDATE = 10
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 4
EPSILON = 1
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01


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

        q_values = self.predict(state)
        return np.argmax(q_values[0])

    def predict(self, state: np.array) -> np.array:
        """Predict q-values using the Q_function
        :param state: an array of floats describing the current environment state
        :return: q_values predicted from model which estimates the Q function
        """
        q_values = self.model.predict(state.reshape(1, -1))
        return q_values


class ExperienceReplay:
    """Implementation of a memory buffer - a deque in a fixed size to store past experiences"""

    def __init__(self, size: int) -> None:
        """
        :param size: size of the deque (int)
        """
        self._memory_buffer = deque(maxlen=size)

    def add(
            self,
            state: np.array,
            action: int,
            reward: float,
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


def train_agent(
        agent: DQN,
        target: DQN,
        env,
        gamma: float = DISCOUNT_FACTOR,
        replay_size: int = REPLAY_MEMORY_SIZE,
        batch_size: int = BATCH_SIZE,
        epsilon: float = EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
) -> tuple[list[float], list[int]]:
    """
    :param agent: agent DQN
    :param target: target DQN
    :param env: the environment which the networks are learning
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

    for episode in range(M_EPISODES):
        state = env.reset()
        terminated = False
        episode_reward = 0
        epsilon = max(epsilon * epsilon_decay, EPSILON_MIN)
        loss = []
        steps_since_update = 0

        while not terminated:
            # select action with prob epsilon of being random
            action = agent.sample_action(state, epsilon)
            # execute action in the environment and observe new state and reward
            next_state, reward, terminated, _ = env.step(action)
            episode_reward += reward

            # Store the transition in replay memory
            experience_replay.add(state, action, reward, next_state, terminated)
            # sample a minibatch of transitions from replay memory
            transitions_minibatch = experience_replay.sample_batch(batch_size)
            if not transitions_minibatch:
                continue

            minibatch_states, minibatch_y = [], []
            for state, action, reward, next_state, terminated in transitions_minibatch:
                # if not terminated, get y from target DQN q-values for the next state
                y = reward if terminated else reward + gamma * target.predict(next_state).max()
                minibatch_states.append(state)
                minibatch_y.append(y)
            minibatch_states, minibatch_y = (
                np.array(minibatch_states),
                np.array(minibatch_y),
            )

            # perform gradient decent step on MSE loss (model compiled with MSE)
            history = agent.model.fit(
                x=minibatch_states, y=minibatch_y, batch_size=batch_size
            )
            loss.append(history.history["loss"][0])

            # if steps reached number of steps for update
            steps_since_update += 1
            if steps_since_update == C_STEPS_UPDATE:
                steps_since_update = 0
                target.model.set_weights(agent.model.get_weights())

        print(f"Episode: {episode + 1} | Loss: {loss:.2f} | Reward: {episode_reward:.2f}")
        # when the episode is finished, log the average loss of the episode
        avg_episode_losses.append(sum(loss) / len(loss))
        episode_rewards.append(episode_reward)

    return avg_episode_losses, episode_rewards


# TODO: Implement test_agent + rendering
# def test_agent():


def main():
    # Load environment and check it state and action parameters
    env = gymnasium.make(ENVIRONMENT)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Build two DQN networks, the agent and the "fixed" target
    layers = DQN_HIDDEN_3
    dqn_agent = DQN(n_states, n_actions, lr=LEARNING_RATE, optimizer=Adam, loss=MSE, layers=layers)
    dqn_target = DQN(n_states, n_actions, lr=LEARNING_RATE, optimizer=Adam, loss=MSE, layers=layers)

    train_agent(dqn_agent, dqn_target, env)


if __name__ == "__main__":
    main()
