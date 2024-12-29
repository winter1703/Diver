import time
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm  # Correct import
from .dive import Board

class Simulator:
    def __init__(self, policy, board_kwargs=None):
        self.policy = policy
        self.board_kwargs = board_kwargs or {}
        self.new_board()

    def new_board(self, seed=None):
        self.board = Board(seed=seed, **self.board_kwargs)
        self.state = None

    def step(self):
        possible_boards = [self.board.possible_boards(move) for move in range(4)]
        action, state, q_values = self.policy(
            tiles=self.board.tiles,
            valid_moves=self.board.valid_moves(),
            rewards=self.board.move_reward(),
            state=self.state,
            p_boards=possible_boards
        )
        self.board.act(action)
        self.state = state
        return action, q_values

    def generate_data(self, buffer_size=100000, save_path=None, pbar=None):
        boards = torch.zeros(buffer_size, *self.board.size, dtype=torch.int16)
        q_values = torch.zeros(buffer_size, 4, *self.board.size)

        for i in range(buffer_size):
            boards[i] = torch.Tensor(self.board.tiles.copy())
            _, q = self.step()
            q_values[i] = torch.Tensor(q)

            if self.board.game_over():
                self.new_board()

            if pbar:
                pbar.update(1)  # Update the progress bar

        if save_path:
            torch.save({"boards": boards, "q_values": q_values, "n_vocab": self.board.max_value}, save_path)
        else:
            return boards, q_values

    def play(self, seed=None):
        self.new_board(seed)
        while not self.board.game_over():
            self.step()
        return self.board.score

    def demo(self, interval=0.5):
        self.new_board()
        print(self.board)
        while not self.board.game_over():
            self.step()
            print(self.board)
            time.sleep(interval)

    def get_current_board(self):
        return self.board.tiles.copy()

def batch_generate_data(policy, save_path, num_workers=5, buffer_size=100000, board_kwargs={}):
    def worker(worker_id, policy, buffer_size, board_kwargs, results):
        simulator = Simulator(policy, board_kwargs)
        with tqdm(total=buffer_size, desc=f"Worker {worker_id}", position=worker_id) as pbar:
            boards, q_values = simulator.generate_data(buffer_size=buffer_size, pbar=pbar)
        results[worker_id] = (boards, q_values)

    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    # Calculate the buffer size for each worker
    worker_buffer_size = buffer_size // num_workers

    # Create and start processes
    for i in range(num_workers):
        process = multiprocessing.Process(target=worker, args=(i, policy, worker_buffer_size, board_kwargs, results))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Combine results from all workers
    all_boards = torch.cat([results[i][0] for i in range(num_workers)])
    all_q_values = torch.cat([results[i][1] for i in range(num_workers)])

    n_vocab = board_kwargs.get("max_value", 10000)

    # Save the combined data
    torch.save({"boards": all_boards, "q_values": all_q_values, "n_vocab": n_vocab}, save_path)

    print(f"Data generation complete. Saved to {save_path}")

def compare_policy(policy_dict, n_run=1000, board_kwargs=None):
    results = {}
    for name in policy_dict:
        policy = policy_dict[name]
        sim = Simulator(policy, board_kwargs)
        scores = np.zeros(n_run)
        print(f"Running policy {name} for {n_run} times")
        for i in tqdm(range(n_run)):
            score = sim.play()
            scores[i] = score
        results[name] = scores
        print(f"Max score: {scores.max()}")
        print(f"99th percentile score: {np.percentile(scores, 99):.1f}")
        print(f"90th percentile score: {np.percentile(scores, 90):.1f}")
        print(f"Median score: {np.median(scores)}")
        print(f"Mean score: {scores.mean()}")
    return results