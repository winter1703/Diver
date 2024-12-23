import random
import numpy as np

MOVE_UP = 0
MOVE_DOWN = 2
MOVE_LEFT = 1
MOVE_RIGHT = 3

def slide_row(row: np.ndarray):
    # row: [N]
    row = row.copy()
    moved = False
    score = 0
    pointer = 0
    for i in range(row.size):
        if i == 0 or row[i] == 0:
            continue
        if row[pointer] == 0:
            row[pointer] = row[i]
            row[i] = 0
            moved = True
        else:
            if row[i] % row[pointer] == 0 or row[pointer] % row[i] == 0:
                score += min(row[i], row[pointer])
                row[pointer] = row[pointer] + row[i]
                row[i] = 0
                moved = True
            elif i > pointer + 1:
                row[pointer + 1] = row[i]
                row[i] = 0
                moved = True
            pointer += 1
    return row, moved, score

def slide(tiles: np.ndarray):
    # tiles: [K, N]
    tiles_new = np.zeros_like(tiles)
    moved = False
    score = 0
    for k in range(tiles.shape[0]):
        row, row_moved, row_score = slide_row(tiles[k, :])
        tiles_new[k, :] = row
        moved = moved or row_moved
        score += row_score
    return tiles_new, moved, score

def slide_to(tiles: np.ndarray, move):
        if move == MOVE_UP:
            tiles, moved, score = slide(tiles.T)
            return tiles.T, moved, score
        if move == MOVE_DOWN:
            tiles, moved, score = slide(np.flip(tiles.T, axis=-1))
            return np.flip(tiles, axis=-1).T, moved, score
        if move == MOVE_LEFT:
            tiles, moved, score = slide(tiles)
            return tiles, moved, score
        if move == MOVE_RIGHT:
            tiles, moved, score = slide(np.flip(tiles, axis=-1))
            return np.flip(tiles, axis=-1), moved, score

def scan(tiles: np.ndarray):
    result = []
    for move in range(4):
        result.append(slide_to(tiles, move))
    return result

def zero_tiles(tiles: np.ndarray):
    zero_indices = np.where(tiles == 0)
    return list(zip(zero_indices[0], zero_indices[1]))

class Board:
    def __init__(self,
                 size = (4, 4),
                 tile_spawn = [2, 3, 5, 7],
                 seed = None,
                 invalid_penalty = -10,
                 max_value = 10000,
                 max_bonus = 1000):
        self.size = size
        self.tile_spawn = tile_spawn
        self.rng = random.Random()
        self.max_value = max_value
        self.invalid_penalty = invalid_penalty
        self.new_board(seed)

    def spawn(self):
        empty_tiles = zero_tiles(self.tiles)
        coord = self.rng.choice(empty_tiles)
        new_tile = self.rng.choice(self.tile_spawn)
        self.tiles[coord[0], coord[1]] = new_tile

    def new_board(self, seed=None):
        self.tiles = np.zeros(self.size, dtype=np.int16)
        self.score = 0
        self.turn = 0

        self.seed = seed or random.randint(0, 2**32 - 1)
        self.rng.seed(self.seed)

        # place two initial tiles
        self.spawn()
        self.spawn()
        self.update()

    def load(self, tiles, score=0, turn=0):
        self.tiles = tiles
        self.score = score
        self.turn = turn
        self.update()

    def update(self):
        self.next = scan(self.tiles)
    
    def act(self, move):
        if self.next[move][1]:
            self.tiles = self.next[move][0]
            self.score += self.next[move][2]
            self.turn += 1
            self.spawn()
            self.update()

    def valid_moves(self):
        return [move for move, (_, moved, _) in enumerate(self.next) if moved]
    
    def move_reward(self):
        return np.array([score if moved else self.invalid_penalty for (_, moved, score) in self.next])

    def game_over(self):
        return not any(item[1] for item in self.next) or np.any(self.tiles > self.max_value)

    def possible_value(self):
        vals = []
        for i in range(self.max_value + 1):
            for p in self.tile_spawn:
                if i % p == 0:
                    vals.append(i)
                    break
        return vals
    
    def possible_boards(self, move):
        board = self.next[move][0]
        empty_tiles = zero_tiles(board)
        K, P = len(empty_tiles), len(self.tile_spawn)
        p_boards = np.tile(board[np.newaxis, np.newaxis, :, :], (K, P, 1, 1))
        for k in range(K):
            for p in range(P):
                p_boards[k, p, empty_tiles[k][0], empty_tiles[k][1]] = self.tile_spawn[p]
        return p_boards.reshape(-1, *p_boards.shape[2:])

    def __str__(self):
        s = "=" * 20
        s += f"\nTurn #{self.turn} Score: {self.score}\n"
        s += str(self.tiles)
        s += "\n"
        s += "=" * 20
        return s

if __name__ == "__main__":
    board = Board()
    mapping = {
        "w": MOVE_UP,
        "a": MOVE_LEFT,
        "s": MOVE_DOWN,
        "d": MOVE_RIGHT,
    }
    while not board.game_over():
        print(board)
        move = random.choice(["w", "a", "s", "d"])
        board.act(mapping[move])
    print(board)