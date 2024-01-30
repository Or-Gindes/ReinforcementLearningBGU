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
from dqn import DQN, ExperienceReplay, DeepQLearning, plot_training_graphs, parse_config

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
DISCOUNT_FACTOR = 0.995
BATCH_SIZE = 64
EPSILON = 1
EPSILON_DECAY = 0.9975
EPSILON_MIN = 0.005


class DoubleDQN(DeepQLearning):
    """This class is in charge of executing the DQN algorithm"""

    def __init__(self, env=ENVIRONMENT, dqn_model=DQN, dqn_layers=DQN_HIDDEN_3, lr=LEARNING_RATE,
                 loss=nn.MSELoss, device: torch.device = None) -> None:
        """
        :param env: the environment which the networks are learning
        :param dqn_model: The model used in the algorithm
        :param dqn_layers: how to build the model
        :param lr: learning rate for the model optimizer
        :param loss: compatible loss class
        :param device: "cpu" or "cuda" - device to train on, if not provided default to cpu
        """
        super().__init__(env, dqn_model, dqn_layers, lr, loss, device)

    def _build_dqn(self, optimizer):
        """Build The two required Agent deep q learning networks"""
        # Build two DQN networks
        self.agent_a = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)
        self.agent_b = self.dqn(self.n_states, self.n_actions, layers=self.dqn_layers).to(self.device)

        self.optimizer_a = optimizer(self.agent_a.parameters(), lr=self.lr)
        self.optimizer_b = optimizer(self.agent_b.parameters(), lr=self.lr)

    def sample_action(self, state: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        with probability epsilon select random action otherwise select greedily according to q_values
        :param state: an array of floats describing the current environment state
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :return: the number of the action that should be taken according to decaying ε-greedy
        """
        if np.random.rand() <= epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

        # Select action based on maximal q_value from both agent networks
        with torch.no_grad():
            a_q_values = self.agent_a(state)
            b_q_values = self.agent_b(state)
            q_values = torch.max(a_q_values, b_q_values)
        action = q_values.max(1).indices.view(1, 1)
        return action

    def train_agent(
            self,
            optimizer=optim.Adam,
            episode_count: int = M_EPISODES,
            gamma: float = DISCOUNT_FACTOR,
            replay_size: int = REPLAY_MEMORY_SIZE,
            batch_size: int = BATCH_SIZE,
            epsilon: float = EPSILON,
            epsilon_decay: float = EPSILON_DECAY,
    ) -> Tuple[List[float], List[int], int]:
        """
        :param optimizer: compatible optimizer class
        :param episode_count: Number of episodes to train for
        :param gamma: discount factor
        :param replay_size: size of the experience replay buffer
        :param batch_size: size of batches to be sampled form the experience replay buffer
        :param epsilon: exploration-exploitation tradeoff - chance of selecting a random action
        :param epsilon_decay: rate of epsilon decay
        :return: a tuple of lists. The first, a list of average loss value per episode.
        The second, a list of rewards accumulated per episode
        """
        self._build_dqn(optimizer)
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
                action = self.sample_action(state, epsilon)
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

                # Randomly chose an agent to update
                chosen_agent = random.choice([self.agent_a, self.agent_b])
                target_agent = self.agent_a if chosen_agent is self.agent_b else self.agent_b
                chosen_optimizer = self.optimizer_a if chosen_agent is self.agent_b else self.optimizer_b

                agent_q_values = chosen_agent(minibatch_states).gather(1, minibatch_actions)
                with torch.no_grad():
                    max_next_state_q_values = target_agent(minibatch_next_states).max(1).values

                # if done, reward is zero and there is no next_step for the update
                max_next_state_q_values[minibatch_done] = torch.zeros(sum(minibatch_done),
                                                                      device=self.device)
                other_expected_q_values = minibatch_rewards + (gamma * max_next_state_q_values)

                # perform gradient descent step on MSE loss (model compiled with MSE)
                loss_value = self.loss(agent_q_values, other_expected_q_values.unsqueeze(1))
                chosen_optimizer.zero_grad()
                loss_value.backward()
                chosen_optimizer.step()
                loss.append(loss_value.item())

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

    plot_training_graphs(config_loss, config_rewards, config_names, results_dir='DDQN_graphs')


if __name__ == "__main__":
    main()
