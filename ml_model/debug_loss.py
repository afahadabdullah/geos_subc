import torch
import yaml
from dataset_flow import S2SHybridDataset
from torch.utils.data import DataLoader

with open("ml_model/config_flow.yaml", "r") as f:
    config = yaml.safe_load(f)

ds = S2SHybridDataset(
    data_root=config["data_dir"],
    start_year=2019,
    end_year=2020,
    normalize=True,
    preload=False,
    stats_file="v5_global_stats.pt"
)

print(f"Dataset size: {len(ds)}")
dl = DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(dl))

print("Batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} | Min: {v.min().item():.3f} | Max: {v.max().item():.3f} | Mean: {v.mean().item():.3f}")
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"  !!! WARNING: NaNs or Infs present in {k} !!!")

