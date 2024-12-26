import torch
from torch.utils import data
from torch.utils.data import random_split

class DiveDataset(data.Dataset):
    def __init__(self, data_path, q_scale=0.05):
        data = torch.load(data_path, weights_only=True)
        self.boards = data["boards"]
        self.q_values = data["q_values"]
        self.n_vocab = data["n_vocab"]
        self.q_scale = q_scale

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.q_scale * self.q_values[idx]

    def get_board_statistics(self):
        flattened_boards = self.boards.view(-1)
        unique_values, counts = torch.unique(flattened_boards, return_counts=True)
        statistics = {int(value): int(count) for value, count in zip(unique_values, counts)}
        sorted_statistics = dict(sorted(statistics.items(), key=lambda item: item[1], reverse=True))
        return sorted_statistics

def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, val_split=0.2, q_scale=0.05):
    # Load the dataset
    dataset = DiveDataset(data_path, q_scale)
    
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
    dataset = DiveDataset("data/data_01.pt")
    print("Board Statistics:", dataset.get_board_statistics())
    
    # Example usage of get_dataloader
    train_dataloader, val_dataloader = get_dataloader("data/data_01.pt")
    print("Training batches:", len(train_dataloader))
    print("Validation batches:", len(val_dataloader))
    
    # Check the dtype of data in the first batch of the training dataloader
    train_batch = next(iter(train_dataloader))
    print("\nTraining Batch Data Types:")
    print("Boards dtype:", train_batch[0].dtype)
    print("Q-values dtype:", train_batch[1].dtype)
    
    # Check the dtype of data in the first batch of the validation dataloader
    val_batch = next(iter(val_dataloader))
    print("\nValidation Batch Data Types:")
    print("Boards dtype:", val_batch[0].dtype)
    print("Q-values dtype:", val_batch[1].dtype)
