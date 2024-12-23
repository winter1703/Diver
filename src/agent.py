import random
import numpy as np
from .dive import Board

def max_q_idx(q):
    max_value = max(q)
    max_indices = [i for i, val in enumerate(q) if val == max_value]
    return random.choice(max_indices)

def policy_random(rewards, **kwargs):
    return random.randint(0, 3), None, rewards

def policy_rotate(rewards, state, valid_moves, **kwargs):
    if not state:
        state = 0
    while not state in valid_moves:
        state = (state + 1) % 4
    return state, (state + 1) % 4, rewards
    
def policy_greed(rewards, **kwargs):
    return max_q_idx(rewards), None, rewards


class DQNAgent:
    def __init__(self, model, epsilon=0.05, gamma=0.99):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma

    def policy(self, tiles, rewards, p_boards, **kwargs):
        pass

    def policy_no_search(self, tiles, **kwargs):
        q_value = self.model(tiles)
        if random.random() < self.epsilon:
            return random.randint(), None, q_value
        else:
            return max_q_idx(q_value), None, q_value