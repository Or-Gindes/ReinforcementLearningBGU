"""
This script is an implementation of an agent using the basic DQN algorithm
Reinforcement Learning course - Assignment 1 - Section 2
"""
import os
from typing import Optional, Tuple, List, Union, SupportsFloat, Any
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

ENVIRONMENT = "CartPole-v1"
LEARNING_RATE = 0.001
DQN_HIDDEN_3 = (256, 128, 64)
DQN_HIDDEN_5 = (512, 256, 128, 64, 32)
REPLAY_MEMORY_SIZE = 1e6

M_EPISODES = 5000
C_STEPS_UPDATE = 50
DISCOUNT_FACTOR = 0.995
BATCH_SIZE = 64
EPSILON = 1
EPSILON_DECAY = 0.9975
EPSILON_MIN = 0.005


class DQN(nn.Module):
    """This class implements the DQN neural network"""

    def __init__(
            self,
            num_states: int,
            num_actions: int,
            layers: Tuple[int, ...] = DQN_HIDDEN_3,
    ) -> None:
        """
        :param num_states: number of states possible in the estimated environment
        :param num_actions: number of actions possible in the estimated environment
        :param layers: Tuple of ints representing layer sizes to build the model with
        """
        super(DQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self._build_model(layers)

    def _build_model(self, layers: Tuple[int, ...]) -> None:
        """
        Method to build the DQN model according to layers input
        The model is the approximation of the action-value function Q and is initialized by default with random weights
        :param layers: Tuple of layer sizes to build the model with
        """
        self.input_layer = nn.Linear(self.num_states, layers[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.output_layer = nn.Linear(layers[-1], self.num_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x


class ExperienceReplay:
    """Implementation of a memory buffer - a deque in a fixed size to store past experiences"""

    def __init__(self, size: int) -> None:
        """
        :param size: size of the deque (int)
        """
        self._memory_buffer = deque(maxlen=int(size))

    def add(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            terminated: torch.Tensor,
    ) -> None:
        """
        :param state: current state
        :param action: current action
        :param reward: reward for current (state, action) pair
        :param next_state: state which the current (state, action) pair lead to
        :param terminated: whether the simulation was completed by the action
        """
        self._memory_buffer.append((state, action, reward, next_state, terminated))

    def sample_batch(self, batch_size: int) -> Optional[Tuple[np.array, ...]]:
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

    def __init__(self,
                 env=ENVIRONMENT,
                 dqn_model=DQN,
                 dqn_layers=DQN_HIDDEN_3,
                 lr=LEARNING_RATE,
                 optimizer=optim.Adam,
                 loss=nn.MSELoss):
        """
        :param env: the environment which the networks are learning
        :param dqn_model: The model used in the algorithm
        :param dqn_layers: how to build the model
        :param lr: learning rate for the model optimizer
        :param optimizer: compatible optimizer class
        :param loss: compatible loss class
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load environment and check it state and action parameters
        self.env = gymnasium.make(env)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.dqn = dqn_model
        self.dqn_layers = dqn_layers
        self.lr = lr
        self.optimizer = None
        self.loss = loss

        self._build_dqn(optimizer)

    def _build_dqn(self, optimizer):
        """Build Agent and Target deep q learning networks"""
        # Build two DQN networks, the agent and the "fixed" target
        self.agent = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)
        self.target = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)
        # Verify the Q-function is initialized the same in both agent and target
        self.target.load_state_dict(self.agent.state_dict())
        self.optimizer = optimizer(self.agent.parameters(), lr=self.lr)

    def sample_action(self, state: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        with probability epsilon select random action otherwise select greedily according to q_values
        :param state: an array of floats describing the current environment state
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :return: the number of the action that should be taken according to decaying ε-greedy
        """
        if np.random.rand() <= epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

        with torch.nograd():
            q_values = self.agent(state)
        action = q_values.max(1).indices.view(1, 1)
        return action

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
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            terminated = False
            episode_reward = 0
            epsilon = max(epsilon * epsilon_decay, EPSILON_MIN)
            loss = []
            steps_since_update = 0

            while not terminated:
                # select action with prob epsilon of being random
                action = self.sample_action(state, epsilon)
                # execute action in the environment and observe new state and reward
                next_state, reward, terminated, _, _ = self.env.step(action.item())
                episode_reward += reward
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)
                terminated = torch.tensor([terminated], device=self.device)

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
                    list(zip(*transitions_minibatch)))
                minibatch_states = torch.cat(minibatch_states)
                minibatch_actions = torch.cat(minibatch_actions)
                minibatch_rewards = torch.cat(minibatch_rewards)
                minibatch_next_states = torch.cat(minibatch_next_states)
                terminated_mask = torch.cat(minibatch_terminated)

                agent_q_values = self.agent(minibatch_states).gather(1, minibatch_actions)
                with torch.no_grad():
                    max_next_state_q_values = self.target(minibatch_next_states).max(1).values

                max_next_state_q_values[terminated_mask] = torch.zeros(len(minibatch_terminated), device=self.device)
                expected_q_values = minibatch_rewards + (gamma * max_next_state_q_values)

                # perform gradient decent step on MSE loss (model compiled with MSE)
                loss_value = self.loss(agent_q_values, expected_q_values.unsqueeze(1))
                self.optimizer.zero_grad()
                loss_value.backward()
                # nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                self.optimizer.step()
                loss.append(loss_value.item())

                # if steps reached number of steps for update
                steps_since_update += 1
                if steps_since_update == C_STEPS_UPDATE:
                    steps_since_update = 0
                    self.target.model.load_state_dict(self.agent.model.state_dict())

            if loss:
                avg_episode_loss = sum(loss) / len(loss)
                # when the episode is finished, log the average loss of the episode
                avg_episode_losses.append(avg_episode_loss)
                episode_rewards.append(episode_reward)

                average_reward_last_100 = sum(episode_rewards[-100:]) / 100
                tqdm_episodes.set_postfix(
                    average_episode_loss=avg_episode_loss,
                    episode_reward=episode_reward,
                    average_reward_last_100_episodes=average_reward_last_100
                )
                # Number of episodes until the agent obtains an average reward >= 475 over 100 consecutive episodes
                if not first_threshold_episode and average_reward_last_100 >= 475.0:
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
            action = self.sample_action(state, epsilon)
            # execute action in the environment and observe new state and reward
            next_state, reward, terminated, _, _ = self.env.step(action)
            total_reward += reward
            if render:
                self.env.render()

            state = next_state

        print(f"Score of the trained agent in a new episode = {total_reward}")


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
    plt.plot(range(len(losses)), losses)
    plt.title("DQN - Average MSE Loss per Training Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average MSE Loss")
    plt.savefig(os.path.join(full_path, "Losses_per_Episode"))

    plt.clf()
    # Plot rewards -
    plt.plot(range(len(rewards)), rewards)
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
