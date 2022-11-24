# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import itertools
from functools import partial
import copy
from tqdm import trange
from scipy.special import softmax

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


def phi(s, eta):
    return np.cos(np.pi * np.dot(s, eta))

def fourier_basis(etas):
    base = []
    eta_norms = []
    # etas = np.array(list(itertools.permutations([i for i in range(p+1)], r=2)))
    # etas = np.array([[1, 1],
    #                 [1, 2]])
    for eta in etas:
        base.append(partial(phi, eta=eta))
        eta_norms.append(np.linalg.norm(eta))
    return base, eta_norms


def fourier_basis_eval(base, s):
    return np.array([f(s) for f in base])


def Q_function(s, a, w, base):
    return np.dot(w[a, :], fourier_basis_eval(base, s))

def V_function(s, w, base):
    Q_array = np.array([Q_function(s, a, w, base) for a in range(0, 3)])
    return max(Q_array)

def epsilon_greedy_policy(s, w, base, epsilon):
    random_int = np.random.uniform(0, 1)

    if random_int > epsilon:
        Q_array = np.array([Q_function(s, a, w, base) for a in range(0, 3)])
        action = np.argmax(Q_array)
    else:
        action = env.action_space.sample()

    return action

def optimal_policy(s, w, base):
    Q_array = np.array([Q_function(s, a, w, base) for a in range(0, 3)])
    action = np.argmax(Q_array)
    action = env.action_space.sample()
    return action

def vectorized_optimal_policy(X, Y, w, base):
    Z = np.zeros(shape=(X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            state = np.array([x, y])
            action = epsilon_greedy_policy(state, w, base, 0)
            Z[i, j] = action
    return Z

def vectorized_V(X, Y, w, base):
    Z = np.zeros(shape=(X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            state = np.array([x, y])
            Z[i, j] = V_function(state, w, base)
    return Z

def update_z(z, a_t, lamda, gamma, grad):
    z = copy.deepcopy(z)
    for a in range(z.shape[0]):
        if a == a_t:
            z[a] = gamma * lamda * z[a] + grad
        else:
            z[a] *= gamma * lamda

    return z

def exploration_policy(s, w, base, epsilon):
    random = np.random.uniform(0, 1)

    if random > epsilon:
        Q_array = np.array([Q_function(s, a, w, base) for a in range(0, 3)])
        action = np.argmax(Q_array)
    else:
        random = np.random.uniform(0, 1)
        if random > 0.5:
            action = env.action_space.sample()
        else : 
            action = 0

    return action

def boltzman_policy(s, w, base, temperature):
    Qs = np.array([0, 0, 0])
    
    for a in range(3):
        Qs[a] = Q_function(s, a, w, base)   

    probs = softmax(Qs/temperature)
    action = np.random.choice([0, 1, 2], 1,
                          p = probs)
    return int(action)
    
def alphas_t(alphas, decay_rate, t):
    return (1/(1 + decay_rate * t)) * alphas
    
def epsilon_t(epsilon, decay_rate, t):
    return (1/(1 + decay_rate * t)) * epsilon

def sarsa_lambda(N_episodes, w_, base, epsilon, lamda, m, gamma, alphas, decay_rate):
    env = gym.make('MountainCar-v0')
    episode_reward_list = []
    w = np.copy(w_)
    c = 0 # c will increase if we get closer to the reward
    w_list = []
    for i in range(N_episodes):
        # Reset enviroment data
        t = 0
        z_t = np.zeros(shape=w.shape)
        v = np.zeros(shape=w.shape)
        done = False
        total_episode_reward = 0.

        s_t_min_1 = scale_state_variables(env.reset()[0])

        e_t = epsilon_t(epsilon=epsilon, decay_rate=decay_rate, t=c)
        learning_rates = alphas_t(
                    alphas=alphas, decay_rate=decay_rate, t=c)

        while not done:
            # Select an action according to an eps greedy policy wrt Q
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            # Compute s_t, a_t, r_t, s_t+1, a_t+1
            a_t = epsilon_greedy_policy(s=s_t_min_1, w=w, base=base, epsilon=e_t)
            s_t, r_t, done, _, _= env.step(a_t)
            s_t = scale_state_variables(s_t)

            a_t_1 = epsilon_greedy_policy(s=s_t, w=w, base=base, epsilon=e_t)
            s_t_1, r_t_1, _, _, _ = env.step(a_t_1)
            s_t_1 = scale_state_variables(s_t_1)

            # Update vector z

            grad = fourier_basis_eval(base=base, s=s_t)
            z_t = np.clip(update_z(z=z_t, a_t=a_t, lamda=lamda,
                        gamma=gamma, grad=grad), -5, 5)

            # Update vector w

            # Compute delta

            delta_t = r_t + gamma * \
                Q_function(s=s_t_1, a=a_t_1, w=w, base=base) - \
                Q_function(s=s_t, a=a_t, w=w, base=base)
            # Update v

            v = m * v + learning_rates * delta_t * z_t
            # Update w with SGD

            w = w + m * v + learning_rates * delta_t * z_t 

            # Update state for next iteration

            s_t_min_1 = s_t

            #Update t 
            t += 1
            total_episode_reward += r_t
            if t > 200:
                done = True
        # Append episode reward
        if total_episode_reward > -85:
            c += 1
        episode_reward_list.append(total_episode_reward)
        w_list.append(w)
        # Close environment
        env.close()
    best_w_ids = np.argmax(episode_reward_list)
    w_best = w_list[best_w_ids]
    return w, episode_reward_list, w_best


def sarsa_lambda_explo(N_episodes, w_, base, epsilon, lamda, m, gamma, alphas, decay_rate):
    env = gym.make('MountainCar-v0')
    episode_reward_list = []
    w = np.copy(w_)
    c = 0 # c will increase if we get closer to the reward
    w_list = []
    for i in range(N_episodes):
        # Reset enviroment data
        t = 0
        z_t = np.zeros(shape=w.shape)
        v = np.zeros(shape=w.shape)
        done = False
        total_episode_reward = 0.

        s_t_min_1 = scale_state_variables(env.reset()[0])

        e_t = epsilon_t(epsilon=epsilon, decay_rate=decay_rate, t=c)
        learning_rates = alphas_t(
                    alphas=alphas, decay_rate=decay_rate, t=c)

        while not done:
            # Select an action according to an eps greedy policy wrt Q
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            # Compute s_t, a_t, r_t, s_t+1, a_t+1
            a_t = exploration_policy(s=s_t_min_1, w=w, base=base, epsilon=e_t)
            s_t, r_t, done, _, _= env.step(a_t)
            s_t = scale_state_variables(s_t)

            a_t_1 = exploration_policy(s=s_t, w=w, base=base, epsilon=e_t)
            s_t_1, r_t_1, _, _, _ = env.step(a_t_1)
            s_t_1 = scale_state_variables(s_t_1)

            # Update vector z

            grad = fourier_basis_eval(base=base, s=s_t)
            z_t = np.clip(update_z(z=z_t, a_t=a_t, lamda=lamda,
                        gamma=gamma, grad=grad), -5, 5)

            # Update vector w

            # Compute delta

            delta_t = r_t + gamma * \
                Q_function(s=s_t_1, a=a_t_1, w=w, base=base) - \
                Q_function(s=s_t, a=a_t, w=w, base=base)
            # Update v

            v = m * v + learning_rates * delta_t * z_t
            # Update w with SGD

            w = w + m * v + learning_rates * delta_t * z_t 

            # Update state for next iteration

            s_t_min_1 = s_t

            #Update t 
            t += 1
            total_episode_reward += r_t
            if t > 200:
                done = True
        # Append episode reward
        if total_episode_reward > -85:
            c += 1
        episode_reward_list.append(total_episode_reward)
        w_list.append(w)
        # Close environment
        env.close()
    best_w_ids = np.argmax(episode_reward_list)
    w_best = w_list[best_w_ids]
    return w, episode_reward_list, w_best




def sarsa_lambda_boltzman(N_episodes, w_, base, temperature, lamda, m, gamma, alphas, decay_rate):
    env = gym.make('MountainCar-v0')
    episode_reward_list = []
    w = np.copy(w_)
    c = 0 # c will increase if we get closer to the reward
    w_list = []
    for i in range(N_episodes):
        # Reset enviroment data
        t = 0
        z_t = np.zeros(shape=w.shape)
        v = np.zeros(shape=w.shape)
        done = False
        total_episode_reward = 0.

        s_t_min_1 = scale_state_variables(env.reset()[0])

        learning_rates = alphas_t(
                    alphas=alphas, decay_rate=decay_rate, t=c)

        while not done:
            # Select an action according to an eps greedy policy wrt Q
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            # Compute s_t, a_t, r_t, s_t+1, a_t+1
            a_t = boltzman_policy(s=s_t_min_1, w=w, base=base, temperature=temperature)
            s_t, r_t, done, _, _= env.step(a_t)
            s_t = scale_state_variables(s_t)

            a_t_1 = boltzman_policy(s=s_t, w=w, base=base, temperature=temperature)
            s_t_1, r_t_1, _, _, _ = env.step(a_t_1)
            s_t_1 = scale_state_variables(s_t_1)

            # Update vector z

            grad = fourier_basis_eval(base=base, s=s_t)
            z_t = np.clip(update_z(z=z_t, a_t=a_t, lamda=lamda,
                        gamma=gamma, grad=grad), -5, 5)

            # Update vector w

            # Compute delta

            delta_t = r_t + gamma * \
                Q_function(s=s_t_1, a=a_t_1, w=w, base=base) - \
                Q_function(s=s_t, a=a_t, w=w, base=base)
            # Update v

            v = m * v + learning_rates * delta_t * z_t
            # Update w with SGD

            w = w + m * v + learning_rates * delta_t * z_t 

            # Update state for next iteration

            s_t_min_1 = s_t

            #Update t 
            t += 1
            total_episode_reward += r_t
            if t > 200:
                done = True
        # Append episode reward
        if total_episode_reward > -85:
            c += 1
        episode_reward_list.append(total_episode_reward)
        w_list.append(w)
        # Close environment
        env.close()
    best_w_ids = np.argmax(episode_reward_list)
    w_best = w_list[best_w_ids]
    return w, episode_reward_list, w_best

def evaluate_policy(w, base, N_episodes):
    env = gym.make('MountainCar-v0')
    episode_reward_list = []
    print('Checking solution...')
    episodes = trange(N_episodes, desc='Episode: ', leave=True)

    for i in episodes:
        episodes.set_description("Episode {}".format(i))

        # Reset enviroment data
        t = 0
        done = False
        total_episode_reward = 0.

        s_t_min_1 = scale_state_variables(env.reset()[0])

        e_t = 0

        while not done:
            # Select an action according to an eps greedy policy wrt Q
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            # Compute s_t, a_t, r_t, s_t+1, a_t+1
            a_t = epsilon_greedy_policy(s=s_t_min_1, w=w, base=base, epsilon=e_t)
            s_t, r_t, done, _, _= env.step(a_t)
            s_t = scale_state_variables(s_t)

            a_t_1 = epsilon_greedy_policy(s=s_t, w=w, base=base, epsilon=e_t)
            s_t_1, r_t_1, _, _, _ = env.step(a_t_1)
            s_t_1 = scale_state_variables(s_t_1)

            # Update state for next iteration

            s_t_min_1 = s_t

            #Update t 
            t += 1
            total_episode_reward += r_t

            if t > 200:
                done = True
        # Append episode reward

        episode_reward_list.append(total_episode_reward)
        # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_episodes)
    return avg_reward, confidence



# --------------------------------------------------------- For training -------------------------------------------------------------------------------


# Parameters
N_episodes = 200    # Number of episodes to run for training
p = 2
gamma = 1.
lamda = 0.9
epsilon = 0.1

# Fourier base and vectors

etas = np.array([
       [0, 1],
        [1, 0],
        [2, 1],
        [1, 2]])

size = etas.shape[0]
base, eta_norms = fourier_basis(etas)

w = np.array([[0.00811316, 0.00945548, 0.00613721, 0.00278877],
        [0.00807464, 0.0071452 , 0.00016493, 0.00356401],
       [0.00774367, 0.00117103, 0.00183444, 0.00852109]])

# Learning rate and m

m = 0.95

alpha = 0.00667
alphas = np.array([alpha/eta_norms[i] for i in range(size)])
decay_rate = 0.2

# w_, episode_reward_list, w_best = sarsa_lambda_boltzman(N_episodes=N_episodes, w_=w, base=base,
#                                                 temperature=0.5, lamda=lamda, m=m, gamma=gamma, alphas=alphas, decay_rate=decay_rate)
# avg_reward, confidence = evaluate_policy(w=w_best, N_episodes=50, base=base)
