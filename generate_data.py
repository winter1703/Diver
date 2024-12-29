from src.simulate import batch_generate_data
from src.agent import get_policy_search, policy_rotate

policy = policy_rotate

data_path = "data/grid_easy_test_100k.pt"

board_kwargs = {
    "max_value": 1000
}

batch_generate_data(policy, data_path, num_workers=8, buffer_size=100000, board_kwargs=board_kwargs)