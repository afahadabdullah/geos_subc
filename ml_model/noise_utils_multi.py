import torch

import noise_utils


def generate_dynamic_multimodal_noise_multi(
    batch,
    E,
    device,
    pr_mjo_bases,
    pr_nao_bases,
    pr_enso_bases,
    t2m_mjo_bases,
    t2m_nao_bases,
    t2m_enso_bases,
    nao_lookup,
    oni_lookup,
    mjo_df,
    flow_matcher,
    year,
    use_lhs=False,
    t2m_random_only=False,
):
    """Generate 2-channel [PR, T2M] dynamic multimodal noise using per-variable EOF bases."""
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    H, W = batch['x_obs'].shape[-2:]

    pr_noise = noise_utils.generate_dynamic_multimodal_noise(
        batch,
        E,
        device,
        pr_mjo_bases,
        pr_nao_bases,
        nao_lookup,
        pr_enso_bases,
        oni_lookup,
        mjo_df,
        flow_matcher,
        year,
        use_lhs=use_lhs,
    )

    if t2m_random_only:
        t2m_noise = torch.randn((vB * E, 1, H, W), device=device)
    else:
        t2m_noise = noise_utils.generate_dynamic_multimodal_noise(
            batch,
            E,
            device,
            t2m_mjo_bases,
            t2m_nao_bases,
            nao_lookup,
            t2m_enso_bases,
            oni_lookup,
            mjo_df,
            flow_matcher,
            year,
            use_lhs=use_lhs,
        )

    if use_lhs:
        pr_noise = noise_utils.orthogonalize_noise_batch(pr_noise, vB, E)
        t2m_noise = noise_utils.orthogonalize_noise_batch(t2m_noise, vB, E)

    return torch.cat([pr_noise, t2m_noise], dim=1)


def print_noise_channel_stats(noise_tensor, prefix="Noise"):
    """Diagnostic stats for each noise channel."""
    print(f"    📊 [{prefix}] Shape: {list(noise_tensor.shape)}")
    for c in range(noise_tensor.shape[1]):
        ch = noise_tensor[:, c]
        print(
            f"       Ch{c}: Mean={ch.mean().item():.4f}, Std={ch.std().item():.4f}, "
            f"Min={ch.min().item():.4f}, Max={ch.max().item():.4f}"
        )
