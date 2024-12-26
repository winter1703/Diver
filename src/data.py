import torch
from torch.utils import data
from torch.utils.data import random_split

class DiveDataset(data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.boards = data["boards"]
        self.q_values = data["q_values"]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.q_values[idx]

    def get_board_statistics(self):
        flattened_boards = self.boards.view(-1)
        unique_values, counts = torch.unique(flattened_boards, return_counts=True)
        statistics = {int(value): int(count) for value, count in zip(unique_values, counts)}
        sorted_statistics = dict(sorted(statistics.items(), key=lambda item: item[1], reverse=True))
        return sorted_statistics

def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, val_split=0.2):
    # Load the dataset
    dataset = DiveDataset(data_path)
    
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
    # Example usage
    train_dataloader, val_dataloader = get_dataloader("data/data_01.pt")
    print("Training batches:", len(train_dataloader))
    print("Validation batches:", len(val_dataloader))
