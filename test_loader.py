from data_loader import data_provider

# --------------------------------------------------
# Fake args object (simple container)
# --------------------------------------------------
class Args:
    root_path = "../electricity_dataset"     # folder where csv is
    data_path = "electricity.csv"

    seq_len = 96
    label_len = 48
    pred_len = 96

    batch_size = 32
    num_workers = 0


args = Args()

# --------------------------------------------------
# LOAD TRAIN LOADER
# --------------------------------------------------
train_dataset, train_loader = data_provider(args, flag='train')

# --------------------------------------------------
# CHECK ONE BATCH
# --------------------------------------------------
for batch_x, batch_y in train_loader:
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)
    print("dataset length:", len(train_dataset))
    print("x min/max:", batch_x.min().item(), batch_x.max().item())
    print("y min/max:", batch_y.min().item(), batch_y.max().item())
    break