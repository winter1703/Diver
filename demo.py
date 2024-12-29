from src.agent import get_policy_search
from src.simulate import Simulator
from UI.game import Game

sim = Simulator(get_policy_search(), board_kwargs={"tile_spawn": [2, 3, 5, 7]})

game = Game(sim)
game.run(interval=0.4)