import os

from envs.tictactoe import TicTacToe
from agents.q_agent import QAgent

EPISODES = 50000


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def act(positions):
        while True:
            matrix = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                      4: (1, 0), 5: (1, 1), 6: (1, 2),
                      7: (2, 0), 8: (2, 1), 9: (2, 2)}
            user_input = int(input("Input your action 1-9:"))
            pos = matrix[user_input]
            if pos in positions:
                return pos


def train():
    p1 = QAgent("p1")
    p2 = QAgent("p2")
    agents = [p1, p2]
    env = TicTacToe(p1, p2)
    for i in range(EPISODES):
        if i % 1000 == 0:
            print("Episode {}".format(i))
        env.reset()
        done = False
        while not done:
            for agent in agents:
                if not done:
                    ava_positions = env.ava_positions()
                    action = agent.act(ava_positions, env.board, env.player_symbol)
                    state, reward, done, info = env.step(action)
                    agent.add_state(agent.convert_board(state))
                    if done:
                        if reward == 1:
                            p1.feed_reward(1)
                            p2.feed_reward(0)
                        elif reward == -1:
                            p1.feed_reward(0)
                            p2.feed_reward(1)
                        else:
                            p1.feed_reward(.1)
                            p2.feed_reward(.5)
                        p1.reset()
                        p2.reset()
        env.reset()
    p1.save_policy()


def play():
    p1 = QAgent("p1")
    p1.load_policy("policy_p1")
    p2 = HumanPlayer("p2")
    agents = [p1, p2]
    env = TicTacToe(p1, p2)
    env.reset()
    done = False
    env.render()
    while not done:
        for agent in agents:
            if agent == p1:
                action = agent.act(env.ava_positions(), env.board, env.player_symbol)
            else:
                action = agent.act(env.ava_positions())
            state, reward, done, info = env.step(action)
            env.render()
            if done:
                print(info['result'])
                break


if __name__ == '__main__':
    if not os.path.exists("policy_p1"):
        train()
    play()
