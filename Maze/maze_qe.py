import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    ACTIONS = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]
    gamma = 1/30
    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_REWARD = -200

    # state
    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_actions_min = 4
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        """The states are of the form (x, y, x_m, y_m, u_t)

        Returns:
            _type_: _description_
        """
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i, j] != 1:
                            # The Minotaur can go everywhere, m is False if alive True if dead
                            states[s] = (i, j, k, l)
                            map[(i, j, k, l)] = s
                            s += 1
        return states, map

    def __move(self, state, action, action_min_n):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used or the agent is dead, the agent stays in place.

            :return tuple next_cell: Position (x,y, x_m, y_m) on the maze that agent transitions to.
        """

        player_state = self.states[state][0], self.states[state][1]
        minotaur_state = self.states[state][2], self.states[state][3]
        # We select one action with uniform probability
        action_min = self.actions[self.ACTIONS[action_min_n]]
        action = self.actions[action]

        # Compute the future position given current (state, action)
        row = self.states[state][0] + action[0]
        col = self.states[state][1] + action[1]

        row_min = self.states[state][2] + action_min[0]
        col_min = self.states[state][3] + action_min[1]

        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1]) or \
            (self.maze[row, col] == 1)

        hitting_maze_walls_minotaur = (row_min == -1) or (row_min == self.maze.shape[0]) or (
            col_min == -1) or (col_min == self.maze.shape[1])

        while hitting_maze_walls_minotaur:  # We have to update the Minotaur if the action chosen for it is
            # We select one action with uniform probability
            action_min = self.actions[self.ACTIONS[np.random.randint(0, 4)]]
            row_min = self.states[state][2] + action_min[0]
            col_min = self.states[state][3] + action_min[1]
            hitting_maze_walls_minotaur = (row_min == -1) or (row_min == self.maze.shape[0]) or (
                col_min == -1) or (col_min == self.maze.shape[1])
        # Based on the impossiblity check return the next state.

        # row_min, col_min = 5, 6
        if hitting_maze_walls:
            state = (self.states[state][0],
                     self.states[state][1], row_min, col_min)
            return self.map[state]

        # No one moves if game is one
        if self.maze[player_state[0], player_state[1]] == 2:
            return state

        # If the game did not stop it continues
        else:
            state = (row, col, row_min, col_min)
            return self.map[state]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for a_min in range(self.n_actions_min):
                    next_s = self.__move(s, a, a_min)
                    transition_probabilities[next_s, s, a] = 1

                    x, y, x_m, y_m = self.states[s]
                    if x == x_m and y == y_m:
                        transition_probabilities[next_s, s, a] = 0

                    if self.maze[x, y] == 2:
                        transition_probabilities[next_s, s, a] = 0
                        
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    for a_min in range(self.n_actions_min):
                        next_s = self.__move(s, a, a_min)
                        x, y, x_m, y_m = self.states[next_s]
                        # Rewrd for hitting a wall
                        if s == next_s and a != self.STAY:
                            rewards[s, a] = self.IMPOSSIBLE_REWARD
                        # Reward for the minotaur
                        if (x, y) == (x_m, y_m):
                            rewards[s, a] = self.MINOTAUR_REWARD
                        # Reward for reaching the exit
                        elif s == next_s and self.maze[x, y] == 2:
                            rewards[s, a] = self.GOAL_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            rewards[s, a] = self.STEP_REWARD

                        # If there exists trapped cells with probability 0.5
                        if random_rewards and self.maze[self.states[next_s]] < 0:
                            row, col = self.states[next_s]
                            # With probability 0.5 the reward is
                            r1 = (1 + abs(self.maze[row, col])) * rewards[s, a]
                            # With probability 0.5 the reward is
                            r2 = rewards[s, a]
                            # The average reward
                            rewards[s, a] = 0.5*r1 + 0.5*r2
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    for a in range(self.a_min):
                        next_s = self.__move(s, a, a_min)
                        i, j = self.states[next_s]
                        # Simply put the reward as the weights o the next state.
                        rewards[s, a] = weights[i][j]

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        # if method == 'DynProg':
        #     # Deduce the horizon from the policy shape
        #     horizon = policy.shape[1]
        #     # Initialize current state and time
        #     t = 0
        #     s = self.map[start]
        #     # Add the starting position in the maze to the path
        #     path.append(start)
        #     while t < horizon-1:
        #         state = path[t]
        #         player_state = (state[0], state[1])
        #         minotaur_state = (state[2], state[3])
        #         if player_state == minotaur_state:
        #             return path
        #         # Move to next state given the policy and the current state
        #         # pick a random action for min
        #         a_min = np.random.randint(0, 4)
        #         next_s = self.__move(s, int(policy[s, t]), a_min)
        #         # Add the position in the maze corresponding to the next state
        #         # to the path
        #         path.append(self.states[next_s])
        #         # Update time and state for next iteration
        #         t += 1
        #         s = next_s

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state, check if eaten
            threshold = 1 - self.gamma
            a_min = np.random.randint(0, 4)
            next_s = self.__move(s, policy[s], a_min)
            # Add the position in the maze corresponding to the next state to the path except if game over
            if decision(threshold=threshold):
                return path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state and check if eaten
                state = path[t-1]
                player_state = (state[0], state[1])
                minotaur_state = (state[2], state[3])
                if player_state == minotaur_state:
                    return path
                s = next_s
                # Move to next state given the policy and the current state
                a_min = np.random.randint(0, 4)
                next_s = self.__move(s, policy[s], a_min)
                path.append(self.states[next_s])

                if decision(threshold=threshold):
                    return path
                # Add the position in the maze corresponding to the next state
                # to the path
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        #print('The states are :')
        # print(self.states)
        print('The actions are:')
        print(self.actions)
        #print('The mapping of the states:')
        # print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def analyze_policy(path, maze):
    for i in range(len(path)):
        state = path[i]
        player_state = (state[0], state[1])
        minotaur_state = (state[2], state[3])

        if i > 0:
            if maze[state[0], state[1]]==2:
                return "HURRA"

            if player_state == minotaur_state:
                # Clean the board
                return "DEAD"
            
        
    return "POISON"

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        state = path[i]
        player_state = (state[0], state[1])
        minotaur_state = (state[2], state[3])

        grid.get_celld()[player_state].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[player_state].get_text().set_text('Player')
        grid.get_celld()[minotaur_state].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[minotaur_state].get_text().set_text('Minotaur')

        if i > 0:

            if maze[player_state[0], player_state[1]] == 2:
                previous_state = path[i-1]
                previous_player_state = (previous_state[0], previous_state[1])
                previous_minotaur_state = (
                    previous_state[2], previous_state[3])
                grid.get_celld()[previous_player_state
                                 ].set_facecolor(col_map[0])
                grid.get_celld()[previous_player_state].get_text().set_text(
                    "Player at t=" + str(i))
                grid.get_celld()[previous_minotaur_state
                                 ].set_facecolor(col_map[0])
                grid.get_celld()[previous_minotaur_state].get_text().set_text(
                    "Minotaur at t=" + str(i))

                grid.get_celld()[player_state].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[player_state].get_text().set_text(
                    'Player is out')

            if player_state == minotaur_state:
                # Clean the board
                previous_state = path[i-1]
                previous_player_state = (previous_state[0], previous_state[1])
                previous_minotaur_state = (
                    previous_state[2], previous_state[3])
                grid.get_celld()[previous_player_state
                                 ].set_facecolor(col_map[0])
                grid.get_celld()[previous_player_state].get_text().set_text(
                    "Player at t=" + str(i))
                grid.get_celld()[previous_minotaur_state
                                 ].set_facecolor(col_map[0])
                grid.get_celld()[previous_minotaur_state].get_text().set_text(
                    "Minotaur at t=" + str(i))

                #Player is dead

                grid.get_celld()[player_state].set_facecolor(LIGHT_RED)
                grid.get_celld()[player_state].get_text().set_text(
                    'Player is dead')

            else:
                previous_state = path[i-1]
                previous_player_state = (previous_state[0], previous_state[1])
                previous_minotaur_state = (
                    previous_state[2], previous_state[3])
                grid.get_celld()[previous_player_state
                                 ].set_facecolor(col_map[0])
                grid.get_celld()[previous_player_state].get_text().set_text("")
                grid.get_celld()[previous_minotaur_state
                                 ].set_facecolor(col_map[maze[previous_minotaur_state]])
                grid.get_celld()[
                    previous_minotaur_state].get_text().set_text("")

                grid.get_celld()[player_state].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[player_state].get_text().set_text('Player')
                grid.get_celld()[minotaur_state].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[minotaur_state].get_text().set_text(
                    'Minotaur')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def draw_path(maze, path):

    previous_player_states = []
    previous_minotaur_states = []

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title(
        'Policy maximizing the probability of leaving the maze alive - leaving as fast as possible')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(len(path)):

        state = path[i]
        player_state = (state[0], state[1])
        minotaur_state = (state[2], state[3])

        if i == 0:
            grid.get_celld()[player_state].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[minotaur_state].set_facecolor(LIGHT_PURPLE)
            previous_player_states.append(player_state)
            previous_minotaur_states.append(minotaur_state)

        if i > 0:
            previous_player_states.append(player_state)
            previous_minotaur_states.append(minotaur_state)

            if player_state == minotaur_state:
                death_draw(state=path[i], previous_state=path[i-1], grid=grid,
                           previous_minotaur_states=previous_minotaur_states, previous_player_states=previous_player_states, i=i)
                return "dead"

            else:
                draw_state_and_previous_state(state=path[i], previous_state=path[i-1], grid=grid, i=i)
                
                if i > 1:
                    if maze[player_state[0], player_state[1]] == 2:
                        victory_draw(state=path[i], previous_state=path[i-1], grid=grid,
                                    previous_minotaur_states=previous_minotaur_states, i=i)
                        actualize_text(previous_state=path[i-1], grid=grid,
                             previous_minotaur_states=previous_minotaur_states, previous_player_states=previous_player_states)
                        return "victory"

                    actualize_text(previous_state=path[i-1], grid=grid,
                             previous_minotaur_states=previous_minotaur_states, previous_player_states=previous_player_states)
    fig.show()
    return "poison"



def victory_draw(state, previous_state, grid, previous_minotaur_states, i):
    grid.set_fontsize(40)

    # We get the current and previous states

    player_state = (state[0], state[1])
    minotaur_state = (state[2], state[3])

    previous_player_state = (previous_state[0], previous_state[1])
    previous_minotaur_state = (previous_state[2], previous_state[3])

    # Colour the Minotaur and
    grid.get_celld()[previous_minotaur_state].get_text().set_text(
        "M at t=" + str(i-1))
    grid.get_celld()[player_state].set_facecolor(LIGHT_GREEN)
    grid.get_celld()[player_state].get_text().set_text("Player is out")

    if player_state in previous_minotaur_states:
        j = [j for j, val in enumerate(
            previous_minotaur_states) if val == player_state]
        grid.get_celld()[player_state].get_text().set_text(
            "Player is out \n" + "at t = " + str(i) + "\n" "M at t=" + str(j)[1:-1])


def death_draw(state, previous_state, grid, previous_minotaur_states, previous_player_states, i):
    grid.set_fontsize(40)

    # Clean the board

    player_state = (state[0], state[1])
    minotaur_state = (state[2], state[3])

    #Player is dead

    actualize_text(previous_state, grid, previous_minotaur_states, previous_player_states)
    grid.get_celld()[player_state].set_facecolor(LIGHT_RED)
    grid.get_celld()[player_state].get_text().set_text(
        'Player is dead')


def draw_state_and_previous_state(state, previous_state, grid, i):
    grid.set_fontsize(40)
    
    player_state = (state[0], state[1])
    minotaur_state = (state[2], state[3])

    previous_player_state = (previous_state[0], previous_state[1])
    previous_minotaur_state = (previous_state[2], previous_state[3])

    previous_player_state = (previous_state[0], previous_state[1])
    previous_minotaur_state = (previous_state[2], previous_state[3]
                                )

    grid.get_celld()[previous_player_state].get_text().set_text(
        "A at t=" + str(i-1))
    grid.get_celld()[previous_minotaur_state].get_text().set_text(
        "M at t=" + str(i-1))

    grid.get_celld()[player_state].set_facecolor(LIGHT_ORANGE)
    grid.get_celld()[player_state].get_text().set_text(
        "A at t=" + str(i))
    grid.get_celld()[minotaur_state].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[minotaur_state].get_text().set_text(
        "M at t=" + str(i))

def actualize_text(previous_state, grid, previous_minotaur_states, previous_player_states):
    grid.set_fontsize(10)

    previous_player_state = (
        previous_state[0], previous_state[1])
    previous_minotaur_state = (
        previous_state[2], previous_state[3])
    # print(previous_minotaur_state, previous_player_state)
    # print(previous_player_states)

    j = [j for j, val in enumerate(
        previous_player_states) if val == previous_player_state]
    grid.get_celld()[previous_player_state].get_text().set_text(
        "A at t=" + str(j)[1:-1])

    j = [j for j, val in enumerate(
        previous_minotaur_states) if val == previous_minotaur_state]
    grid.get_celld()[previous_minotaur_state].get_text().set_text(
        "M at t=" + str(j)[1:-1])

    if previous_player_state in previous_minotaur_states:
        j0 = [j for j, val in enumerate(
            previous_player_states) if val == previous_player_state]
        j1 = [j for j, val in enumerate(
            previous_minotaur_states) if val == previous_player_state]
        text = "A at t=" + \
            str(j0)[1:-1] + "\n" + "M at t=" + str(j1)[1:-1]
        grid.get_celld()[
            previous_player_state].get_text().set_text(text)

    if previous_minotaur_state in previous_player_states:
        j0 = [j for j, val in enumerate(
            previous_player_states) if val == previous_minotaur_state]
        j1 = [j for j, val in enumerate(
            previous_minotaur_states) if val == previous_minotaur_state]
        text = "A at t=" + \
            str(j0)[1:-1] + "\n" + "M at t=" + str(j1)[1:-1]
        grid.get_celld()[
            previous_minotaur_state].get_text().set_text(text)

def decision(threshold):
    return np.random.random() < threshold
