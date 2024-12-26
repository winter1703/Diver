import torch
import torch.nn.functional as F
from .model import DiveModel
from .embed import DiveEmbed
from .dive import Board

class DiveTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, device="cuda"):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for boards, q_values in self.train_dataloader:
            boards = boards.to(self.device)
            q_values = q_values.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(boards)
            loss = F.mse_loss(outputs, q_values)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for boards, q_values in self.val_dataloader:
                boards = boards.to(self.device)
                q_values = q_values.to(self.device)
                
                outputs = self.model(boards)
                loss = F.mse_loss(outputs, q_values)
                total_loss += loss.item()
        return total_loss / len(self.val_dataloader)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
