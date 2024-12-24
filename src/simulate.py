import time
import numpy as np
import tqdm
from .dive import Board

class Simulator:
    def __init__(self, policy):
        self.policy = policy

    def step(self, board: Board, state):
        possible_boards = [board.possible_boards(move) for move in range(4)]
        action, state, q_values = self.policy(tiles=board.tiles,
                                  valid_moves=board.valid_moves(),
                                  rewards=board.move_reward(),
                                  state=state,
                                  p_boards=possible_boards)
        board.act(action)
        return action, state, q_values

    def play(self):
        board = Board()
        seed = board.seed
        state = None # optional memory slot for policy
        actions = []
        train_data = []

        while not board.game_over():
            action, state, q_values = self.step(board, state)
            actions.append(action)
            train_data.append((board.tiles, q_values))
            
        final_score = board.score
        return seed, actions, final_score, train_data
    
    def demo(self, interval=0.5):
        board = Board()
        state = None
        print(board)
        while not board.game_over():
            _, state, _ = self.step(board, state)
            print(board)
            time.sleep(interval)

def compare_policy(policy_dict, n_run=1000):
    results = {}
    for name in policy_dict:
        policy = policy_dict[name]
        sim = Simulator(policy)
        scores = np.zeros(n_run)
        print(f"Running policy {name} for {n_run} times")
        for i in tqdm.tqdm(range(n_run)):
            _, _, score, _ = sim.play()
            scores[i] = score
        results[name] = scores
        print(f"Max score: {scores.max()}")
        print(f"99th percentile score: {np.percentile(scores, 99):.1f}")
        print(f"90th percentile score: {np.percentile(scores, 90):.1f}")
        print(f"Median score: {np.median(scores)}")
        print(f"Mean score: {scores.mean()}")
    return results