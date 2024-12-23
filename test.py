from src.agent import policy_random, policy_rotate, policy_greed
from src.simulate import compare_policy

policy_dict = {
    "Random": policy_random,
    "Rotate": policy_rotate,
    "Greedy": policy_greed,
}

compare_policy(policy_dict)
