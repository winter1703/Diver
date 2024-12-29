import torch
import numpy as np
from src.model import DiveModel
from src.data import DiveDataset

MODEL_PATH = "checkpoint/model_g_easy_4b50.ckpt"
DATA_PATH = "data/grid_easy_test_100k.pt"

# Initialize the model and dataset
model = DiveModel(d_embed=64, d_vocab=1000, n_block=4)
dataset = DiveDataset(DATA_PATH)

checkpoint = torch.load(MODEL_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Randomly select 1 indices from the dataset
indices = torch.randint(0, len(dataset), (1,))

# Compare model output with ground truth for each selected index
with torch.no_grad():
    for idx in indices:
        board, q_value = dataset[idx]
        board = board.unsqueeze(0).to("cpu")  # Add batch dimension and move to CPU
        output = model(board)  # Get model prediction
        
        # Print the results with formatted output to .2f
        print(f"Index: {idx}")
        print("Board:\n", board.squeeze().numpy())  # Remove batch dimension, convert to numpy, and round to 2 decimal places
        
        # Scale back to original range and round to 2 decimal places
        q_value_scaled = np.round(q_value.numpy() / dataset.q_scale, 2)
        output_scaled = np.round(output.squeeze().numpy() / dataset.q_scale, 2)
        # Sum over the last two dimensions (board size)
        q_value_sum = np.sum(q_value_scaled, axis=(-2, -1))
        output_sum = np.round(np.sum(output_scaled, axis=(-2, -1)), 2)
        
        print("Ground Truth Q-value:\n", q_value_scaled)
        print("Model Output Q-value:\n", output_scaled)
        
        print("Ground Truth Q-value Sum:\n", q_value_sum)
        print("Model Output Q-value Sum:\n", output_sum)
        print()
