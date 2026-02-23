import torch
from diffusion import ConditionalDiffusion

def test_sampler():
    # Setup dummy model and inputs
    device = "cpu"
    model = ConditionalDiffusion(
        in_channels=4, condition_channels=49, out_channels=4,
        block_out_channels=(64, 128, 256), layers_per_block=1
    ).to(device)
    
    # Dummy condition
    condition = torch.randn(2, 49, 181, 360, device=device)
    
    # Sample
    model.eval()
    samples = model.sample(condition, num_inference_steps=50, verbose=True)
    
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {samples.mean().item():.4f}")
    print(f"Sample std: {samples.std().item():.4f}")
    print(f"Sample min: {samples.min().item():.4f}")
    print(f"Sample max: {samples.max().item():.4f}")

if __name__ == "__main__":
    test_sampler()
