import torch

import noise_utils


def _get_batch_scalar(values, idx, default=None):
    """Read a scalar from a batch field that may be a tensor, list, or scalar."""
    if values is None:
        return default
    if isinstance(values, torch.Tensor):
        if values.ndim == 0:
            return values.item()
        return values[idx].item()
    if isinstance(values, (list, tuple)):
        return values[idx]
    return values


def _resolve_sample_init_date(batch, b_idx, fallback_year):
    """Prefer the exact sample init date when the dataset provides it."""
    year = _get_batch_scalar(batch.get('year'), b_idx, fallback_year)
    month = _get_batch_scalar(batch.get('month'), b_idx)
    day = _get_batch_scalar(batch.get('day'), b_idx, 15)
    return noise_utils.datetime.date(int(year), int(month), int(day))


def _generate_single_channel_dynamic_noise(
    batch,
    E,
    device,
    mjo_bases,
    nao_bases,
    enso_bases,
    nao_lookup,
    oni_lookup,
    mjo_df,
    year,
    use_lhs=False,
):
    """Generate one structured noise channel without calling the multi-channel EOF sampler."""
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    H, W = batch['x_obs'].shape[-2:]
    init_dates = [_resolve_sample_init_date(batch, b_idx, year) for b_idx in range(vB)]

    pure_noise = torch.randn((vB * E, 1, H, W), device=device)

    mjo_noise = torch.randn((vB * E, 1, H, W), device=device)
    if mjo_bases is not None:
        mjo = batch.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if not isinstance(mjo, torch.Tensor):
            mjo = torch.tensor(mjo, dtype=torch.long)
        lead = batch['lead_idx'].clone().detach() if isinstance(batch['lead_idx'], torch.Tensor) else torch.tensor(batch['lead_idx'])

        for b_idx in range(vB):
            phase = int(mjo[b_idx])
            lead_idx = int(lead[b_idx])
            if not use_lhs:
                for j in range(E):
                    mjo_noise[b_idx * E + j, 0] = noise_utils.sample_from_eof_basis(
                        mjo_bases, phase, lead_idx, device, H, W
                    )
            else:
                mjo_noise[b_idx * E:(b_idx + 1) * E, 0] = noise_utils.sample_batch_lhs(
                    mjo_bases, phase, lead_idx, device, H, W, E
                )

    nao_noise = torch.randn((vB * E, 1, H, W), device=device)
    if nao_bases is not None and nao_lookup is not None:
        leads = batch['lead_idx']
        for b_idx in range(vB):
            lead_idx = int(_get_batch_scalar(leads, b_idx))
            init_date = init_dates[b_idx]
            nao_phase = noise_utils.get_nao_phase(init_date, nao_lookup)

            if not use_lhs:
                for j in range(E):
                    nao_noise[b_idx * E + j, 0] = noise_utils.sample_from_eof_basis(
                        nao_bases, nao_phase, lead_idx, device, H, W
                    )
            else:
                nao_noise[b_idx * E:(b_idx + 1) * E, 0] = noise_utils.sample_batch_lhs(
                    nao_bases, nao_phase, lead_idx, device, H, W, E
                )

    enso_noise = torch.randn((vB * E, 1, H, W), device=device)
    if enso_bases is not None and oni_lookup is not None:
        leads = batch['lead_idx']
        for b_idx in range(vB):
            init_date = init_dates[b_idx]
            lead_idx = int(_get_batch_scalar(leads, b_idx))
            enso_state = noise_utils.get_enso_state(init_date.month, init_date.year, oni_lookup)

            if not use_lhs:
                for j in range(E):
                    enso_noise[b_idx * E + j, 0] = noise_utils.sample_from_eof_basis(
                        enso_bases, enso_state, lead_idx, device, H, W
                    )
            else:
                enso_noise[b_idx * E:(b_idx + 1) * E, 0] = noise_utils.sample_batch_lhs(
                    enso_bases, enso_state, lead_idx, device, H, W, E
                )

    blend = torch.empty_like(pure_noise)
    for b_idx, init_date in enumerate(init_dates):
        start = b_idx * E
        end = start + E

        mjo_amp = 1.0
        date_str = init_date.strftime('%Y-%m-%d')
        if mjo_df is not None and date_str in mjo_df.index:
            row = mjo_df.loc[date_str]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                row = row.iloc[0]
            r1 = row.get('RMM1_lagged', 0.0)
            r2 = row.get('RMM2_lagged', 0.0)
            if not (noise_utils.pd.isna(r1) or noise_utils.pd.isna(r2)):
                mjo_amp = float(noise_utils.np.sqrt(r1**2 + r2**2))

        nao_amp = 0.5
        if nao_lookup is not None:
            nao_amp = abs(noise_utils.get_nao_value(init_date, nao_lookup))

        enso_amp = 0.5
        if oni_lookup is not None:
            enso_amp = abs(noise_utils.get_enso_value(init_date.month, init_date.year, oni_lookup))

        mjo_amp = max(min(mjo_amp, 3.0), 0.1)
        nao_amp = max(min(nao_amp, 2.5), 0.1)
        enso_amp = max(min(enso_amp, 2.5), 0.1)

        total = mjo_amp + nao_amp + enso_amp
        w_mjo = mjo_amp / total
        w_nao = nao_amp / total
        w_enso = enso_amp / total

        blend[start:end] = (
            0.90
            * (
                w_mjo * mjo_noise[start:end]
                + w_nao * nao_noise[start:end]
                + w_enso * enso_noise[start:end]
            )
            + 0.10 * pure_noise[start:end]
        )

    std = blend.std(dim=(2, 3), keepdim=True)
    return blend / (std + 1e-6)


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
    year,
    use_lhs=False,
    t2m_random_only=False,
    orthogonalize_lhs=True,
    pr_random_blend=0.0,
):
    """Generate 2-channel [PR, T2M] dynamic multimodal noise using per-variable EOF bases."""
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    H, W = batch['x_obs'].shape[-2:]

    pr_noise = _generate_single_channel_dynamic_noise(
        batch,
        E,
        device,
        pr_mjo_bases,
        pr_nao_bases,
        pr_enso_bases,
        nao_lookup,
        oni_lookup,
        mjo_df,
        year,
        use_lhs=use_lhs,
    )

    if pr_random_blend > 0.0:
        pr_random = torch.randn((vB * E, 1, H, W), device=device)
        pr_noise = (1.0 - pr_random_blend) * pr_noise + pr_random_blend * pr_random
        pr_std = pr_noise.std(dim=(2, 3), keepdim=True)
        pr_noise = pr_noise / (pr_std + 1e-6)

    if t2m_random_only:
        t2m_noise = torch.randn((vB * E, 1, H, W), device=device)
    else:
        t2m_noise = _generate_single_channel_dynamic_noise(
            batch,
            E,
            device,
            t2m_mjo_bases,
            t2m_nao_bases,
            t2m_enso_bases,
            nao_lookup,
            oni_lookup,
            mjo_df,
            year,
            use_lhs=use_lhs,
        )

    if use_lhs and orthogonalize_lhs:
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
