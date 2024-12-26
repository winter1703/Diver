import torch
from src.training import DiveTrainer
from src.model import DiveModel
from src.data import get_dataloader
from src.dive import Board

model = DiveModel(d_embed=32, d_vocab=10000)

# Calculate and print the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

DATA_PATH = "data/data_01.pt"

data_train, data_val = get_dataloader(DATA_PATH)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

trainer = DiveTrainer(model, data_train, data_val, optimizer)

trainer.train(50)

trainer.save_checkpoint("checkpoint/model_test.ckpt")