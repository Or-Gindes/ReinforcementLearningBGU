"""
This script is an implementation of an agent using the basic DQN algorithm
Reinforcement Learning course - Assignment 1 - Section 2
"""
import os
import json
from typing import Tuple, List
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from torch import Tensor
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

RUN_TYPE = "base"  # "full
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
            done: torch.Tensor,
    ) -> None:
        """
        :param state: current state
        :param action: current action
        :param reward: reward for current (state, action) pair
        :param next_state: state which the current (state, action) pair lead to
        :param done: whether the simulation was completed by the action
        """
        self._memory_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: int) -> list[Tensor] | None:
        """
        :param batch_size: int - size of batch to sample from the memory buffer
        :return: if there are enough samples to return (i.e. more than batch_size) sample randomly and return
        """
        if len(self._memory_buffer) < batch_size:
            return

        sample = random.sample(self._memory_buffer, batch_size)
        return [torch.cat(category) for category in list(zip(*sample))]


class DeepQLearning:
    """This class is in charge of executing the DQN algorithm"""

    def __init__(
            self,
            env=ENVIRONMENT,
            dqn_model=DQN,
            dqn_layers=DQN_HIDDEN_3,
            lr=LEARNING_RATE,
            optimizer=optim.Adam,
            loss=nn.MSELoss,
            device: torch.device = None
    ) -> None:
        """
        :param env: the environment which the networks are learning
        :param dqn_model: The model used in the algorithm
        :param dqn_layers: how to build the model
        :param lr: learning rate for the model optimizer
        :param optimizer: compatible optimizer class
        :param loss: compatible loss class
        :param device: "cpu" or "cuda" - device to train on, if not provided default to cpu
        """
        self.device = device if device else torch.device("cpu")
        # Load environment and check it state and action parameters
        self.env = gymnasium.make(env)
        self.render_env = gymnasium.make(env, render_mode="human")
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.dqn = dqn_model
        self.dqn_layers = dqn_layers
        self.lr = lr
        self.optimizer = None
        self.loss = loss()

        self._build_dqn(optimizer)

    def _build_dqn(self, optimizer):
        """Build Agent and Target deep q learning networks"""
        # Build two DQN networks, the agent and the "fixed" target
        self.agent_a = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)
        self.agent_b = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)
        # Verify the Q-function is initialized the same in both agent and target
        self.agent_b.load_state_dict(self.agent_a.state_dict())
        self.optimizer = optimizer(self.agent_a.parameters(), lr=self.lr)

    def sample_action(self, state: torch.Tensor, epsilon: float, agent) -> torch.Tensor:
        """
        with probability epsilon select random action otherwise select greedily according to q_values
        :param state: an array of floats describing the current environment state
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :return: the number of the action that should be taken according to decaying ε-greedy
        """
        if np.random.rand() <= epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

        with torch.no_grad():
            q_values = agent(state)
        action = q_values.max(1).indices.view(1, 1)
        return action

    def train_agent(
            self,
            episode_count: int = M_EPISODES,
            gamma: float = DISCOUNT_FACTOR,
            replay_size: int = REPLAY_MEMORY_SIZE,
            batch_size: int = BATCH_SIZE,
            epsilon: float = EPSILON,
            epsilon_decay: float = EPSILON_DECAY,
    ) -> Tuple[List[float], List[int], int]:
        """
        :param episode_count: Number of episodes to train for
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

        tqdm_episodes = tqdm(range(0, episode_count), desc="Episode:")
        for episode in tqdm_episodes:
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = False
            episode_reward = 0
            epsilon = max(epsilon * epsilon_decay, EPSILON_MIN)
            loss = []
            steps_since_update = 0

            while not done:
                # select action with prob epsilon of being random

                chosen_agent = random.choice([self.agent_a, self.agent_b])
                other_agent = self.agent_a if chosen_agent == self.agent_b else self.agent_b

                action = self.sample_action(state, epsilon,chosen_agent)
                # execute action in the environment and observe new state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                # end of an episode is reached when (terminated or truncated)
                done = terminated or truncated
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)
                done = torch.tensor([done], device=self.device)

                # Store the transition in replay memory
                experience_replay.add(state, action, reward, next_state, done)

                # After the transition is stored, update current state to the next state S_t+1 = S_t
                state = next_state

                # sample a minibatch of transitions from replay memory
                transitions_minibatch = experience_replay.sample_batch(batch_size)
                if not transitions_minibatch:
                    continue

                # Unpack the minibatch into separate tensors
                (
                    minibatch_states,
                    minibatch_actions,
                    minibatch_rewards,
                    minibatch_next_states,
                    minibatch_done,
                ) = transitions_minibatch

                agent_q_values = chosen_agent(minibatch_states).gather(1, minibatch_actions)
                with torch.no_grad():
                    max_next_state_q_values = other_agent(minibatch_next_states).max(1).values

                max_next_state_q_values[minibatch_done] = torch.zeros(sum(minibatch_done),
                                                                      device=self.device)
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
                    self.agent_b.load_state_dict(self.agent_a.state_dict())

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
                    break

        if first_threshold_episode:
            print(f"Number of episodes to achieve average reward "
                  f"of at least 475 over 100 consecutive episodes: {first_threshold_episode}")

        self.env.close()
        return avg_episode_losses, episode_rewards, first_threshold_episode

    def test_agent(self, render: bool = True) -> int:
        """
        Test the trained agent in env on a new episode and render the environment
        :param render: boolean parameter indicating whether to render the agent playing the environment
        """
        env = self.render_env if render else self.env
        state, _ = env.reset()
        # The model is trained and doesn't require any more exploration when selecting actions in the environment
        epsilon = 0
        done = False
        total_reward = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # select action
            action = self.sample_action(state, epsilon)
            # execute action in the environment and observe new state and reward
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated

            state = next_state
            if total_reward > 1000:
                break
        env.close()
        return total_reward
        # print(f"Score of the trained agent in a new episode = {total_reward}")


def plot_training_graphs(
        losses_list: List[List[float]],
        rewards_list: List[List[int]],
        iteration_names: List[str],
        results_dir: str = "D_DQN_graphs",
):
    """
    Function plots the average losses and rewards recorded per training episode for multiple iterations
    :param losses_list: List of lists, where each inner list contains average losses per training episode for an iteration
    :param rewards_list: List of lists, where each inner list contains total rewards per training episode for an iteration
    :param iteration_names: List of names for each training iteration
    :param results_dir: Folder to save plots in
    """
    full_path = os.path.join(os.getcwd(), results_dir)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    figsize = (12, 8)
    # Plot losses
    plt.figure(figsize=figsize)
    for i, losses in enumerate(losses_list):
        plt.plot(range(len(losses)), losses, label=iteration_names[i])

        plt.title("DQN - Average MSE Loss per Training Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Average MSE Loss")
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.savefig(os.path.join(full_path, f"Losses_per_Episode_{iteration_names[i]}.png"))
        plt.clf()

    # Plot rewards
    plt.figure(figsize=figsize)
    for i, rewards in enumerate(rewards_list):
        plt.plot(range(len(rewards)), rewards, label=iteration_names[i])

        plt.title("DQN - Total Reward per Training Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.savefig(os.path.join(full_path, f"Rewards_per_Episode_{iteration_names[i]}.png.png"))
        plt.clf()


def parse_config(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)

    return config


def main():
    # Set training device to GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU device detected, training on GPU")
    else:
        device = torch.device("cpu")
        print("Couldn't detect GPU, defaulting to CPU for training")

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dqn_configs.json"
    )
    config = parse_config(config_path)
    config_loss = []
    config_rewards = []
    config_names = []
    if RUN_TYPE == "base":
        hyperparameter_scenarios = {"base config": config["training_hyperparams"]["base"]}
    else:
        hyperparameter_scenarios = config["training_hyperparams"]

    n_configs = len(hyperparameter_scenarios.keys())
    for config_number, (conf_name, conf_params) in enumerate(hyperparameter_scenarios.items()):
        print(f"\nTesting config {config_number + 1} of {n_configs}")
        for architecture_name, layers in config["dqn_layers"].items():
            dqn = DeepQLearning(
                env=config["environment"],
                dqn_model=DQN,
                dqn_layers=layers,
                lr=conf_params["learning_rate"],
                optimizer=optim.Adam,
                loss=nn.MSELoss,
                device=device
            )
            avg_episode_losses, episode_rewards, convergence_episode = dqn.train_agent(
                episode_count=config["m_episodes"],
                gamma=conf_params["discount_factor"],
                replay_size=config["replay_memory_size"],
                batch_size=conf_params["batch_size"],
                epsilon=conf_params["epsilon"],
                epsilon_decay=conf_params["epsilon_decay"],
            )
            config_loss.append(avg_episode_losses)
            config_rewards.append(episode_rewards)
            test_reward = dqn.test_agent(render=True)
            config_names.append(
                "__".join(
                    [
                        architecture_name,
                        conf_name,
                        "config",
                        f"test_reward={test_reward if test_reward <= 1000 else '> 1000'}",
                        f"convergence_episode={convergence_episode}",
                    ]
                )
            )

    plot_training_graphs(config_loss, config_rewards, config_names)


if __name__ == "__main__":
    main()
