import pickle

import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3


class QAgent:
    def __init__(self, name, exp_rate=.3):
        self.name = name
        self.states = []  # save all taken positions
        self.lr = .2
        self.exp_rate = exp_rate
        self.decay_gamma = .9
        self.states_value = {}

    @staticmethod
    def convert_board(board):
        return str(board.reshape(BOARD_COLS * BOARD_ROWS))

    def act(self, postions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(postions))
            action = postions[idx]
        else:
            action = None
            value_max = -999
            for p in postions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board_vector = self.convert_board(next_board)
                action_value = 0 if self.states_value.get(next_board_vector) is None else self.states_value.get(
                    next_board_vector)
                if action_value >= value_max:
                    value_max = action_value
                    action = p
        return action

    def add_state(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feed_reward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def save_policy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()
