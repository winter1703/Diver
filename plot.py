from src.data import DiveDataset, plot_board_statistics, plot_zipf_law

DATA_PATH = "data/data_4m.pt"

dataset = DiveDataset(DATA_PATH)

plot_board_statistics(dataset)
plot_zipf_law(dataset)