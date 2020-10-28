import logging

import gym
from gym import spaces
import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3


class TicTacToe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, p1, p2):
        self.action_space = spaces.Discrete(BOARD_COLS * BOARD_ROWS)
        self.observation_space = spaces.Discrete(BOARD_COLS * BOARD_ROWS)
        self.board = np.zeros((BOARD_COLS, BOARD_ROWS))
        self.p1 = p1
        self.p2 = p2
        self.board_vector = None
        self.done = False
        self.seed()
        self.reset()
        # init p1 go first
        self.player_symbol = 1

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board_vector = None
        self.done = False
        self.player_symbol = 1
        return self.board

    def convert_board(self):
        self.board_vector = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.board_vector

    def ava_positions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def update_state(self, position):
        self.board[position] = self.player_symbol
        self.player_symbol = -1 if self.player_symbol == 1 else 1

    def step(self, action):
        # assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self.board, 0, True, None

        self.update_state(loc)
        status = self.check_game_status()
        if status is not None:
            if status == 1:
                reward = 1
                info = {"result": "player 1 win"}
            elif status == -1:
                reward = -1
                info = {"result": "player 2 win"}
            else:
                reward = 0
                info = {"result": "tie"}
            return self.board, reward, self.done, info
        return self.board, None, self.done, None

    def render(self, mode='human'):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                token = ''
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

    def check_game_status(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.done = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.done = True
                return -1

        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.done = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.done = True
                return -1

        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.done = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        if len(self.ava_positions()) == 0:
            self.done = True
            return 0

        self.done = False
        return None
