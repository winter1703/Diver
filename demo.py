from src.agent import policy_random, policy_rotate, policy_greed, model_scan, DQNAgent
from src.simulate import Simulator, compare_policy

search_agent = DQNAgent(model_scan, epsilon=0.0)

policy_dict = {
    "Random": policy_random,
    "Rotate": policy_rotate,
    "Greedy": policy_greed,
    "Search1": search_agent.policy
}

# sim = Simulator(search_agent.policy)
# sim.demo(interval=0.0)

compare_policy(policy_dict, 100)
