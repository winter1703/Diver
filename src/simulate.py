import numpy as np
import tqdm
from .dive import Board

class Simulator:
    def __init__(self, policy):
        self.policy = policy

    def play(self):
        board = Board()
        seed = board.seed
        state = None # optional memory slot for policy
        actions = []
        train_data = []

        while not board.game_over():
            tiles = board.tiles
    
            possible_boards = [board.possible_boards(move) for move in range(4)]
            action, state, q_values = self.policy(tiles=tiles,
                                                  valid_moves=board.valid_moves(),
                                                  rewards=board.move_reward(),
                                                  state=state,
                                                  p_boards=possible_boards)
            actions.append(action)
            board.act(action)

            train_data.append((tiles, q_values))
            
        final_score = board.score
        return seed, actions, final_score, train_data

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
        print(f"Mean score: {scores.mean()}")
        print(f"Median score: {np.median(scores)}")
    return results