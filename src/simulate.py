from dive import Board

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
            valid_moves = board.valid_moves()
            rewards = board.move_reward()
            action, state, q_values = self.policy(tiles, valid_moves, rewards, state)

            actions.append(action)
            board.act(action)

            train_data.append((tiles, rewards, q_values))
            
        final_score = board.score
        return seed, actions, final_score, train_data
