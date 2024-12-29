import torch
import numpy as np
from src.dive import Board
from src.model import DiveModel
import math

# Function to factorize a number
def factorize(n):
    factors = []
    # Check for divisibility by 2
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    # Check for odd factors from 3 to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n = n // i
    # If n is still greater than 2, it's a prime number
    if n > 2:
        factors.append(n)
    return factors

MODEL_PATH = "checkpoint/model_g_easy_500.ckpt"

# Initialize the model and dataset
model = DiveModel(d_embed=64, d_vocab=1000, n_block=2)

checkpoint = torch.load(MODEL_PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

num_list = Board(max_value=1000).possible_value()
embed = model.embed.weight[num_list]

# Function to find the top-K closest vectors and their distances
def find_top_k_closest(embed, target_idx, k=5):
    # Compute L2 distances between the target vector and all other vectors
    distances = torch.norm(embed - embed[target_idx], dim=1)
    
    # Get the indices and values of the top-K closest vectors (excluding the target itself)
    top_k_distances, top_k_indices = torch.topk(distances, k+1, largest=False)
    
    # Exclude the target itself (distance 0)
    return top_k_indices[1:], top_k_distances[1:]

# Example number to search
number = 105  # Replace with your desired number

# Find the index of the number in num_list
target_idx = num_list.index(number)

# Find the top-K closest vectors and their distances
top_k_indices, top_k_distances = find_top_k_closest(embed, target_idx, k=10)

# Print the corresponding numbers and their distances
print(f"Top-10 closest numbers to {number}:")
for idx, dist in zip(top_k_indices, top_k_distances):
    num = num_list[idx]
    factors = factorize(num)
    print(f"Number: {num}, Distance: {dist.item():.4f}, Factors: {factors}")

