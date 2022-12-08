import numpy as np
import gym
from collections import deque
from tqdm import trange
from DQN_agent import Agent
import torch


def running_average(x, N):
    """Function used to compute the running average
    of the last N elements of a vector x
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1 :] = np.convolve(x, np.ones((N,)) / N, mode="valid")
    else:
        y = np.zeros_like(x)
    return y


def dqn(
    N_episodes,
    discount_factor,
    n_ep_running_average,
    batch_size,
    buffer_size,
    alpha,
    epsilon_max,
    epsilon_min,
    seed,
    device
):

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make("LunarLander-v2")
    env.reset()

    # Parameters
    n_actions = env.action_space.n  # Number of available actions
    n_states = len(env.observation_space.high)  # State dimensionality

    # We will use these variables to compute the average episodic reward and the average number of steps per episode

    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode
    episodes_avg = deque(maxlen=50)

    # Random agent initialization and count initialization

    agent = Agent(
        n_actions=n_actions,
        n_states=n_states,
        seed=seed,
        alpha=alpha,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device
    )
    count = 1
    # Training

    EPISODES = trange(N_episodes, desc="Episode: ", leave=True, miniters=1)

    for i in EPISODES:

        # Update epsilon

        epsilon = agent.epsilon_t(
            episode_number=i,
            N_episodes=N_episodes,
            epsilon_max=epsilon_max,
            epsilon_min=epsilon_min,
        )

        # Reset enviroment data and initialize variables
        t = 1

        done = False
        state = torch.tensor(env.reset(), requires_grad=False, dtype=torch.float32)
        total_episode_reward = 0.0

        while not done:

            action = agent.action(state=state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                discount_factor=discount_factor,
                count=count,
            )
            state = torch.tensor(next_state, requires_grad=True, dtype=torch.float32)
            total_episode_reward += reward
            t += 1
            count += 1

        episode_reward_list.append(total_episode_reward)
        episodes_avg.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # if np.mean(episodes_avg) >= 200.:
        #     print(
        #         "\nSolved in {:d} episodes with an \tAverage Score: {:.2f}".format(
        #             i, np.mean(episodes_avg)
        #         )
        #     )
        #     torch.save(agent.main_qnetwork, "neural-network-1.pth")

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i,
                total_episode_reward,
                t,
                np.mean(episodes_avg),
                np.mean(running_average(episode_number_of_steps, n_ep_running_average)),
            )
        )
    # Close environment
    env.close()
    return episode_reward_list, episode_number_of_steps
