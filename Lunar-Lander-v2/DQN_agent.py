# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
from typing import Tuple
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random


class Agent(object):
    """Base agent class, used as a parent class

    Args:
        n_actions (int): number of actions

    Attributes:
        n_actions (int): where we store the number of actions
        last_action (int): last action taken by the agent
    """

    def __init__(
        self,
        n_actions: int,
        n_states: int,
        seed: int,
        alpha: float,
        buffer_size: int,
        batch_size: int,
        device
    ) -> None:

        # General parameters

        self.n_actions = n_actions
        self.n_states = n_states
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )
        self.update_frequency = int(self.buffer_size / self.batch_size)
        self.hidden_size = 64
        self.device = device
        # Networks

        self.main_qnetwork = NeuralNetwork(
            input_size=n_states, hidden_size=self.hidden_size, output_size=n_actions, seed=seed
        ).to(self.device)
        self.target_qnetwork = NeuralNetwork(
            input_size=n_states, hidden_size=self.hidden_size, output_size=n_actions, seed=seed
        ).to(self.device)

        self.target_qnetwork.eval()
        self.update_networks()
        # Optimizer

        self.optimizer = optim.Adam(self.main_qnetwork.parameters(), lr=alpha)

        # Buffer

        self.buffer = ExperienceReplayBuffer(maximum_length=buffer_size, seed=seed)

        

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        discount_factor: float,
        count: int,
    ) -> None:

        # Add last experience to the buffer

        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

        # We learn if len(buffer) > batch_size

        if len(self.buffer) > self.batch_size:

            # Sample a batch of self.batch_size
            batch = self.buffer.sample_batch(n=self.batch_size)
            states, actions, rewards, next_states, dones = batch

            # Perform forward propagation

            Q_values = self.main_qnetwork_forward(states=states, actions=actions)
            targets = self.target_qnetwork_forward(
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                discount_factor=discount_factor,
            )

            # Perform backward propagation

            self.backward(targets=targets, Q_values=Q_values, count=count)

    def action(self, state: np.ndarray, epsilon: float) -> int:

        # Epsilon greedy policy selection of the action with the Q_values

        random_ = np.random.uniform(0, 1)
        if random_ < epsilon:
            return random.randint(0, self.n_actions - 1)

        if not torch.is_tensor(state):
            state = torch.tensor(state, requires_grad=True, device=self.device)
        # Get Q_values with eval
        with torch.no_grad():
            Q_values = self.main_qnetwork(state)
        return torch.argmax(Q_values).item()


    def main_qnetwork_forward(
        self,
        states: Tuple,
        actions: Tuple,
    ) -> torch.Tensor:

        # Compute Q values
        self.main_qnetwork.train()
        return self.main_qnetwork(
            torch.tensor(
                torch.stack([torch.tensor(state) for state in states], dim=0),
                requires_grad=True,
                dtype=torch.float32,
            )
        ).gather(
            1, torch.tensor(actions).unsqueeze(1)
        )  # size batch, 1

    @torch.no_grad()
    def target_qnetwork_forward(
        self,
        rewards: Tuple,
        next_states: Tuple,
        dones: Tuple,
        discount_factor: float,
    ) -> torch.Tensor:

        self.target_qnetwork.eval()
        Q_values_prime = self.target_qnetwork(
            torch.tensor(np.asarray(next_states), requires_grad=False, dtype=torch.float32)
        ).detach()
        return torch.tensor(rewards, requires_grad=False ,dtype=torch.float32).unsqueeze(1) + (
            1 - torch.tensor(dones,requires_grad=False, dtype=torch.float32)
        ).unsqueeze(1) * discount_factor * Q_values_prime.max(1)[0].unsqueeze(1)

    def backward(
        self, targets: torch.Tensor, Q_values: torch.Tensor, count: int
    ) -> None:

        # Compute loss function for the main network, clip grad
        loss = nn.functional.mse_loss(targets, Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_qnetwork.parameters(), max_norm=2.0)
        # Perform backward pass (backpropagation)
        self.optimizer.step()

        if count % self.update_frequency == 0:
            self.update_networks()

    def update_networks(self) -> None:
        self.target_qnetwork.load_state_dict(self.main_qnetwork.state_dict())

    @staticmethod
    def epsilon_t(
        episode_number: int,
        N_episodes: int,
        epsilon_max: float,
        epsilon_min: float,
    ) -> float:
        return max(
            epsilon_min,
            epsilon_max
            - (epsilon_max - epsilon_min)
            * (episode_number - 1)
            / (int(N_episodes * 0.9) - 1),
        )


class RandomAgent(Agent):
    """Agent taking actions uniformly at random, child of the class Agent"""

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        """Compute an action uniformly at random across n_actions possible
        choices

        Returns:
            action (int): the random action
        """
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class ExperienceReplayBuffer(object):
    """Class used to store a buffer containing experiences of the RL agent."""

    def __init__(self, maximum_length: int, seed: int):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)
        self.seed = random.seed(seed)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n: int):
        """Function used to sample experiences from the buffer.
        returns 5 lists, each of size n. Returns a list of state, actions,
        rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError("Tried to sample too many elements from the buffer!")

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(len(self.buffer), size=n, replace=False)

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


### Neural Network ###
class NeuralNetwork(nn.Module):
    """Create a feedforward neural network"""

    def __init__(self, input_size: int, hidden_size: int, seed: int, output_size):
        super(NeuralNetwork, self).__init__()
        # Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.activation_layer = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.activation_layer(l1)

        # Compute first hidden layer
        l1 = self.hidden_layer(l1)
        l1 = self.activation_layer(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out


class NeuralNetwork_2(nn.Module):
    """Create a feedforward neural network"""

    def __init__(self, input_size, hidden_size, output_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Parameters

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.activation_layer = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize the weights

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.hidden_layer_1.weight)
        nn.init.xavier_uniform_(self.hidden_layer_2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.activation_layer(l1)

        # Compute first hidden layer
        l1 = self.hidden_layer_1(l1)
        l1 = self.activation_layer(l1)

        # Compute first hidden layer
        l1 = self.hidden_layer_2(l1)
        l1 = self.activation_layer(l1)
        # Compute output layer
        out = self.output_layer(l1)
        return out


def epsilon_greedy_policy(Q_values, epsilon, env):

    random_int = np.random.uniform(0, 1)

    if random_int > epsilon:

        action = Q_values.max(1)[1].item()
    else:
        action = env.action_space.sample()

    return action


# def update_networks(main_network : nn.Module, target_network : nn.Module):
#     target_network_copy = deepcopy(target_network)
#     for main_params, target_params in zip(main_network.parameters(), target_network_copy.parameters()):
#             target_params.data.copy_(main_params)
#     return target_network_copy


def targets_fun(dones, Q_values_prime, discount_factor, rewards):
    return torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) + (
        1 - torch.tensor(dones, dtype=torch.float32)
    ).unsqueeze(1) * discount_factor * Q_values_prime.max(1)[0].unsqueeze(1)
