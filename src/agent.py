import random
import numpy as np
from .dive import scan

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
    return max_q_idx(np.sum(rewards, axis=(-2, -1))), None, rewards

def q_scan(tiles: np.ndarray):
    return np.array([reward for (_, _, reward) in scan(tiles)])

def model_scan(boards: np.ndarray):
    # boards: [K, M, N]
    K, M, N = boards.shape
    q_value = np.zeros((K, 4, M, N))
    for k in range(K):
        q_value[k] = q_scan(boards[k])
    return q_value

def softmax(x, axis=None, temp=1.0):
    x_scaled = x / temp
    x_exp = np.exp(x_scaled - np.max(x_scaled, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

def expected_max_q(pq_value, temp=1.0):
    sums = np.sum(pq_value, axis=(2, 3))
    softmax_probs = softmax(sums, axis=1, temp=temp)
    softmax_probs = softmax_probs[:, :, np.newaxis, np.newaxis]
    weighted_arrays = pq_value * softmax_probs
    weighted_sum = np.sum(weighted_arrays, axis=1)
    result = np.mean(weighted_sum, axis=0)
    return result
    
def get_policy_search(epsilon=0.0):
    search_agent = DQNAgent(model_scan, epsilon=epsilon)
    return search_agent.policy

class DQNAgent:
    def __init__(self, model, epsilon=0.05, gamma=0.99):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma

    def policy(self, tiles, valid_moves, rewards, p_boards, **kwargs):
        M, N = tiles.shape[-2], tiles.shape[-1]
        q_value = np.zeros(4, M, N)
        for move in range(4):
            if move in valid_moves:
                q_move = expected_max_q(self.model(p_boards[move]))
                q_value[move] = rewards[move] + self.gamma * q_move
            else:
                q_value[move] = rewards[move]
        if random.random() < self.epsilon:
            return random.randint(0, 3), None, q_value
        else:
            return max_q_idx(q_value), None, q_value

    def policy_no_search(self, tiles, **kwargs):
        q_value = self.model(tiles)
        if random.random() < self.epsilon:
            return random.randint(0, 3), None, q_value
        else:
            return max_q_idx(q_value), None, q_value