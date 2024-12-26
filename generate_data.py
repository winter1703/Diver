from src.simulate import batch_generate_data
from src.agent import get_policy_search

policy = get_policy_search()

data_path = "data/data_01.pt"

batch_generate_data(policy, data_path)