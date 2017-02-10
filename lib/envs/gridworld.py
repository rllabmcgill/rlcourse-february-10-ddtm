import numpy as np
import sys
import StringIO

from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

GOAL_REWARD = 5
OTHER_REWARDS = [-12, 10]

TERMINAL_STATE = -1


class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from the paper "Double Q-Learning"
    by van Hasselt (2011):

    o o G
    o o o
    S o o

    S is the agent's starting position and G is the goal state.

    The agent can take actions in each direction
    (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave the agent in its current state.
    Each non-terminating step, the agent receives a random reward of -12 or +10
    with equal probability.
    In the goal state every action yields +5 and ends an episode.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = [3, 3]

        num_states = np.prod(self.shape)
        num_actions = 4

        MAX_Y = self.shape[0]
        MAX_X = self.shape[1]

        P = {}
        grid = np.arange(num_states).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        # Fill the transitions.
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in xrange(num_actions)}

            is_terminal = s == TERMINAL_STATE
            is_goal = s == 2

            if is_terminal:
                P[s][UP] = [(1.0, TERMINAL_STATE, 0, True)]
                P[s][RIGHT] = [(1.0, TERMINAL_STATE, 0, True)]
                P[s][DOWN] = [(1.0, TERMINAL_STATE, 0, True)]
                P[s][LEFT] = [(1.0, TERMINAL_STATE, 0, True)]
            elif is_goal:  # The agent reached the goal state.
                P[s][UP] = [(1.0, TERMINAL_STATE, GOAL_REWARD, True)]
                P[s][RIGHT] = [(1.0, TERMINAL_STATE, GOAL_REWARD, True)]
                P[s][DOWN] = [(1.0, TERMINAL_STATE, GOAL_REWARD, True)]
                P[s][LEFT] = [(1.0, TERMINAL_STATE, GOAL_REWARD, True)]
            else:  # Not a goal state.
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(0.5, ns_up, r, False) for r in OTHER_REWARDS]
                P[s][RIGHT] = [(0.5, ns_right, r, False) for r in OTHER_REWARDS]
                P[s][DOWN] = [(0.5, ns_down, r, False) for r in OTHER_REWARDS]
                P[s][LEFT] = [(0.5, ns_left, r, False) for r in OTHER_REWARDS]

            it.iternext()

        # Initial state distribution is a delta-function at the
        # bottom-left corner.
        s_distribution = np.zeros(num_states)
        s_distribution[6] = 1.0

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(
            num_states, num_actions, P, s_distribution)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            is_goal = s == 2

            if self.s == s:
                output = " x "
            elif is_goal:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
