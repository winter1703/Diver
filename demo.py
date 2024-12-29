from src.agent import policy_random, policy_rotate, policy_greed, model_scan, DQNAgent
from src.simulate import Simulator, compare_policy

search_agent = DQNAgent(model_scan, epsilon=0.0)

policy_dict = {
    "Random": policy_random,
    "Rotate": policy_rotate,
    "Greedy": policy_greed,
    "Search1": search_agent.policy
}

sim = Simulator(search_agent.policy, board_kwargs={"tile_spawn": [2, 3, 5, 7]})


import time

start_time = time.time()
sim.demo(interval=0.0)
end_time = time.time()

run_time = end_time - start_time
print(f"Run time: {run_time:.2f} seconds")

# compare_policy(policy_dict, 100)
