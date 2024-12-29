import matplotlib.pyplot as plt
from src.agent import policy_random, policy_rotate, policy_greed, get_policy_search
from src.simulate import compare_policy

policy_dict = {
    "Random": policy_random,
    "Rotate": policy_rotate,
    "Greedy": policy_greed,
    "Search": get_policy_search()
}

# Run the simulation and get the results
result = compare_policy(policy_dict, 100)

# Create the "chance to score at least" plot
plt.figure(figsize=(10, 6))

# Plot curves for each policy
for name, scores in result.items():
    # Sort the scores
    sorted_scores = sorted(scores)
    # Calculate the percentiles based on the sorted scores
    n = len(sorted_scores)
    percentiles = [(i / (n - 1)) * 100 for i in range(n)]  # Map to percentiles (0 to 100)
    # Plot the sorted scores against (100 - percentiles) to get "chance to score at least"
    plt.plot(sorted_scores, [100 - p for p in percentiles], label=name)

# Set log scale for the x-axis (score axis)
plt.xscale('log')
plt.xlabel('Score (log scale)')
plt.ylabel('Chance to Score at Least (%)')
plt.title('Chance to Score at Least Across Different Policies')
plt.legend()
plt.grid(True)
plt.show()