from src.simulate import batch_generate_data
from src.agent import get_policy_search

policy = get_policy_search()

data_path = "data/data_4m.pt"

batch_generate_data(policy, data_path, num_workers=8, buffer_size=4000000)