import torch
import yaml
from torch.utils.data import DataLoader
from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher
from train_flow import run_val_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('ml_model/config_flow.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Loading dataset...")
val_dataset = S2SHybridDataset(data_root=config['data_dir'], start_year=2021, end_year=2021,
                                normalize=True, preload=False, stats_file='v5_global_stats.pt', subsample_monthly=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Loading model...")
model = FlowMatchingModel(in_channels=35, out_channels=1).to(device)
ckpt = torch.load('ml_output_flow4/latest_flow_ckpt.pt', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model'])
model.eval()

flow_matcher = CustomFlowMatcher(device=device)
eof_bases = torch.load('ml_model/mjo_eof_bases.pt', map_location='cpu')['eof_bases']
from train_flow import area_weights, global_bounds, target_sqrt_max, target_sqrt_min, geos_min, geos_max

print("\n--- Evaluate WITH apply_flow_variance=True ---")
out_true = run_val_inference(116, model, val_loader, flow_matcher, device, None, '.', 'dummy.csv',
                             target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights.to(device),
                             global_bounds, is_test=False, is_fast_recon=True, use_flow_variance=True, eof_bases=eof_bases)

print(f"CRPS WITH VarHead: {out_true[0]:.4f}")

print("\n--- Evaluate WITHOUT apply_flow_variance=False ---")
out_false = run_val_inference(116, model, val_loader, flow_matcher, device, None, '.', 'dummy.csv',
                              target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights.to(device),
                              global_bounds, is_test=False, is_fast_recon=True, use_flow_variance=False, eof_bases=eof_bases)

print(f"CRPS WITHOUT VarHead: {out_false[0]:.4f}")
