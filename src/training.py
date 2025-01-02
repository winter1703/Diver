from tqdm import tqdm
import os, json
import torch
import torch.nn.functional as F
from .data import get_dataloader
from .model import DiveModel

def combined_loss(prediction: torch.Tensor, target: torch.Tensor, eta=1):
    loss = F.mse_loss(prediction, target)
    sum_loss = F.mse_loss(prediction.sum(dim=(-2, -1)), target.sum(dim=(-2, -1)))
    return loss + eta * sum_loss

class DiveTrainer:
    def __init__(self, model, info, train_dataloader, val_dataloader, optimizer, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.info = info
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.info["train_loss_history"] = []
        self.info["val_loss_history"] = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for boards, q_values in tqdm(self.train_dataloader, desc="Training"):
            boards = boards.to(self.device)
            q_values = q_values.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(boards)
            loss = combined_loss(outputs, q_values)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        self.info["train_loss_history"].append(avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for boards, q_values in tqdm(self.val_dataloader, desc="Evaluating"):
                boards = boards.to(self.device)
                q_values = q_values.to(self.device)
                
                outputs = self.model(boards)
                loss = combined_loss(outputs, q_values)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        self.info["val_loss_history"].append(avg_loss)
        return avg_loss

    def save_checkpoint(self, name):
        model_path = os.path.join("checkpoint", name + ".pt")
        # Save the model and optimizer state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

        # Save the training and validation loss history as a JSON file
        info_path = os.path.join("checkpoint", name + ".json")
        with open(info_path, 'w') as f:
            json.dump(self.info, f, indent=4)

    def load_checkpoint(self, name):
        model_path = os.path.join("checkpoint", name + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def train(self, epochs, save_name):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            if self.scheduler:
                self.scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if save_name:
            self.save_checkpoint(save_name)

def config_optimizer(model: torch.nn.Module, optimizer_name, optimizer_kwargs, scheduler_name, scheduler_kwargs):
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    if scheduler_name:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        return optimizer, scheduler
    else:
        return optimizer, None

def config_trainer(dataset_kwargs,
                   model_kwargs,
                   optim_kwargs) -> DiveTrainer:
    dl_train, dl_val, board_kwargs = get_dataloader(dataset_kwargs)
    model = DiveModel(model_kwargs)
    optimizer, scheduler = config_optimizer(model, **optim_kwargs)
    model_card = {
        "board_kwargs": board_kwargs,
        "model_kwargs": model_kwargs,
        "dataset_kwargs": dataset_kwargs,
        "optim_kwargs": optim_kwargs,
    }
    trainer = DiveTrainer(model, model_card, dl_train, dl_val, optimizer, scheduler)
    return trainer

def load_trainer(name):
    info_path = os.path.join("checkpoint", name + ".json")
    with open(info_path, "r") as file:
        info = json.load(file)
    trainer = config_trainer(info["dataset_kwargs"], info["model_kwargs"], info["optim_kwargs"])
    trainer.load_checkpoint(name)
    return trainer