import torch
from torch.utils import data
from torch.utils.data import random_split

class DiveDataset(data.Dataset):
    def __init__(self, data_path, q_scale=0.5, augment=True):
        data = torch.load(data_path, weights_only=True)
        self.boards = data["boards"]
        self.q_values = data["q_values"]
        self.n_vocab = data["n_vocab"]
        self.q_scale = q_scale
        self.augment = augment

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        q_value = self.q_scale * self.q_values[idx]

        if self.augment:
            board, q_value = self._apply_random_transformation(board, q_value)
        
        return board, q_value

    def _apply_random_transformation(self, board, q_value):
        # Randomly select one of the 8 possible transformations
        transform_idx = torch.randint(0, 8, (1,)).item()
        
        # Apply the transformation to the board and q_value
        if transform_idx == 0:
            # Identity (no transformation)
            pass
        elif transform_idx == 1:
            # Rotate 90 degrees clockwise
            board = torch.rot90(board, k=1, dims=[0, 1])
            q_value = torch.rot90(q_value, k=1, dims=[1, 2])
            q_value = q_value[[3, 0, 1, 2], :, :]  # Reorder move dimensions
        elif transform_idx == 2:
            # Rotate 180 degrees
            board = torch.rot90(board, k=2, dims=[0, 1])
            q_value = torch.rot90(q_value, k=2, dims=[1, 2])
            q_value = q_value[[2, 3, 0, 1], :, :]  # Reorder move dimensions
        elif transform_idx == 3:
            # Rotate 270 degrees clockwise
            board = torch.rot90(board, k=3, dims=[0, 1])
            q_value = torch.rot90(q_value, k=3, dims=[1, 2])
            q_value = q_value[[1, 2, 3, 0], :, :]  # Reorder move dimensions
        elif transform_idx == 4:
            # Flip vertically
            board = torch.flip(board, dims=[0])
            q_value = torch.flip(q_value, dims=[1])
            q_value = q_value[[2, 1, 0, 3], :, :]  # Reorder move dimensions
        elif transform_idx == 5:
            # Flip horizontally
            board = torch.flip(board, dims=[1])
            q_value = torch.flip(q_value, dims=[2])
            q_value = q_value[[0, 3, 2, 1], :, :]  # Reorder move dimensions
        elif transform_idx == 6:
            # Flip along the main diagonal
            board = torch.transpose(board, 0, 1)
            q_value = torch.transpose(q_value, 1, 2)
            q_value = q_value[[1, 0, 3, 2], :, :]  # Reorder move dimensions
        elif transform_idx == 7:
            # Flip along the anti-diagonal
            board = torch.rot90(torch.flip(board, dims=[0, 1]), k=1, dims=[0, 1])
            q_value = torch.rot90(torch.flip(q_value, dims=[1, 2]), k=1, dims=[1, 2])
            q_value = q_value[[3, 2, 1, 0], :, :]  # Reorder move dimensions
        
        return board, q_value

    def get_board_statistics(self, sort=True):
        flattened_boards = self.boards.view(-1)
        unique_values, counts = torch.unique(flattened_boards, return_counts=True)
        statistics = {int(value): int(count) for value, count in zip(unique_values, counts)}
        if not sort:
            return statistics
        sorted_statistics = dict(sorted(statistics.items(), key=lambda item: item[1], reverse=True))
        return sorted_statistics
    
    def get_baseline_q_value_mse(self):
        mean_q_value = torch.mean(self.q_values)
        baseline_q_values = torch.full_like(self.q_values, mean_q_value)
        mse_loss = torch.mean((self.q_values - baseline_q_values) ** 2)
        return mean_q_value.item(), (self.q_scale ** 2) * mse_loss.item()

def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, val_split=0.2, q_scale=0.5):
    # Load the dataset
    dataset = DiveDataset(data_path, q_scale, augment=True)
    
    # Calculate the sizes for training and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoader instances for training and validation sets
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Typically, we don't shuffle the validation set
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    # Load the dataset
    dataset = DiveDataset("data/grid_easy_1m.pt")
    
    # Print board statistics
    # print("Board Statistics:", dataset.get_board_statistics(True))
    
    # Print baseline loss
    print("Baseline loss:", dataset.get_baseline_q_value_mse())
    
    # Show 5 random data examples from the dataset
    import random
    print("\n5 Random Data Examples:")
    for _ in range(5):
        idx = random.randint(0, len(dataset) - 1)  # Generate a random index
        board, q_value = dataset[idx]
        print(f"Example {_+1}:")
        print("Board:\n", board.numpy())  # Convert tensor to list for prettier printing
        print("Q-value:", q_value.numpy() / dataset.q_scale)  # Convert scalar tensor to Python float
        print()
