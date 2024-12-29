import torch
from src.training import DiveTrainer
from src.model import DiveModel
from src.data import get_dataloader

model = DiveModel(d_embed=16, d_vocab=1000, n_block=2)

# Calculate and print the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

DATA_PATH = "data/grid_easy_1m.pt"

data_train, data_val = get_dataloader(DATA_PATH, batch_size=2000, val_split=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

trainer = DiveTrainer(model, data_train, data_val, optimizer)

trainer.train(10)

trainer.save_checkpoint("checkpoint/model_baseline.ckpt")