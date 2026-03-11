import torch
import numpy as np
import datetime
import pandas as pd

# ─── Index Parsers ───

def parse_nao_index(nao_path):
    """Parse CPC daily NAO index file."""
    if not __import__('os').path.exists(nao_path): return None
    nao_lookup = {}
    with open(nao_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4: continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            val = float(parts[3])
            d = datetime.date(year, month, day)
            nao_lookup[d] = val
        except ValueError:
            continue
    return nao_lookup

def parse_oni_index(oni_path):
    """Parse CPC ONI file."""
    if not __import__('os').path.exists(oni_path): return None
    oni_lookup = {}
    with open(oni_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4: continue
        try:
            oni_lookup[(int(parts[1]), parts[0].strip())] = float(parts[3])
        except (ValueError, IndexError):
            continue
    return oni_lookup

def get_nao_phase(init_date, nao_lookup, threshold=0.5):
    """Get NAO phase using a 7-day trailing average ending 1 day BEFORE init_date."""
    if isinstance(init_date, pd.Timestamp):
        base_date = init_date.date()
    else:
        base_date = init_date
        
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup:
            vals.append(nao_lookup[d])
            
    if not vals: return 1
    val = sum(vals) / len(vals)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1

def get_nao_value(init_date, nao_lookup):
    if isinstance(init_date, pd.Timestamp):
        base_date = init_date.date()
    else:
        base_date = init_date
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup:
            vals.append(nao_lookup[d])
    if not vals: return 0.0
    return sum(vals) / len(vals)

def get_enso_state(month, year, oni_lookup, threshold=0.5):
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[int(month)]
    lookup_year = year + yr_off
    if int(month) == 1: lookup_year = year - 1
    val = oni_lookup.get((lookup_year, seas), 0.0)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1

def get_enso_value(month, year, oni_lookup):
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[int(month)]
    lookup_year = year + yr_off
    if int(month) == 1: lookup_year = year - 1
    return oni_lookup.get((lookup_year, seas), 0.0)

def sample_from_eof_basis(eof_bases, phase, lead, device, H, W):
    key = (phase, lead)
    if key not in eof_bases: key = phase
    if key not in eof_bases: key = (1, lead)
    if key not in eof_bases: return torch.randn(H, W, device=device)
    
    eofs = eof_bases[key]['eofs'].to(device)
    K = eofs.shape[0]
    alpha = torch.randn(K, device=device)
    noise_field = torch.einsum('k,khw->hw', alpha, eofs)
    std = noise_field.std()
    if std > 1e-6: noise_field = noise_field / std
    return noise_field

def sample_batch_lhs(eof_bases, phase, lead, device, H, W, E):
    key = (phase, lead)
    # Match previous fallbacks
    if key not in eof_bases: key = phase
    if key not in eof_bases: key = (1, lead)
    if key not in eof_bases:
        if isinstance(phase, int) and phase in eof_bases: key = phase
        elif 0 in eof_bases: key = 0
        else: return torch.randn(E, H, W, device=device)
        
    eofs = eof_bases[key]['eofs'].to(device)
    K = eofs.shape[0]
    
    from scipy.stats import qmc, norm
    sampler = qmc.LatinHypercube(d=K)
    sample = sampler.random(n=E)
    alpha = torch.tensor(norm.ppf(sample), dtype=torch.float32, device=device) # [E, K]
    
    if 'eigenvalues' in eof_bases[key]:
        eigenvals = eof_bases[key]['eigenvalues'].to(device)
        alpha = alpha * torch.sqrt(eigenvals)
        
    noise_fields = torch.einsum('ek,khw->ehw', alpha, eofs)
    
    # Normalize each ensemble member
    std = noise_fields.std(dim=(1,2), keepdim=True)
    mask = (std > 1e-6).squeeze()
    if mask.any():
        noise_fields[mask] = noise_fields[mask] / std[mask]
    return noise_fields

def generate_dynamic_multimodal_noise(batch, E, device, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, year, use_lhs=False):
    """
    Generates dynamically weighted multi-modal blended noise for a batch.
    batch: data dict from dataloader
    E: number of ensemble members
    """
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    H, W = batch['x_obs'].shape[-2:]
    
    pure_noise = torch.randn((vB * E, 2, H, W), device=device)
    
    # ── MJO EOFs ──
    mjo_noise = torch.randn((vB * E, 2, H, W), device=device)
    if mjo_bases is not None:
        mjo = batch.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if not isinstance(mjo, torch.Tensor): mjo = torch.tensor(mjo)
        lead = batch['lead_idx'].clone().detach() if isinstance(batch['lead_idx'], torch.Tensor) else torch.tensor(batch['lead_idx'])
        
        if not use_lhs:
            mjo_expanded = mjo.repeat_interleave(E)
            lead_expanded = lead.repeat_interleave(E)
            mjo_noise = flow_matcher.eof_sample(mjo_bases, mjo_expanded, vB * E, H, W, lead_ids=lead_expanded)
        else:
            for b_idx in range(vB):
                p = int(mjo[b_idx])
                l = int(lead[b_idx])
                fields = sample_batch_lhs(mjo_bases, p, l, device, H, W, E)
                mjo_noise[b_idx*E:(b_idx+1)*E, 0] = fields
                mjo_noise[b_idx*E:(b_idx+1)*E, 1] = fields.clone()
    
    # ── NAO EOFs ──
    nao_noise = torch.randn((vB * E, 2, H, W), device=device)
    if nao_bases is not None and nao_lookup is not None:
        months = batch['month']
        leads = batch['lead_idx']
        for b_idx in range(vB):
            m = int(months[b_idx])
            l = int(leads[b_idx])
            init_date = datetime.date(year, m, 15)
            nao_phase = get_nao_phase(init_date, nao_lookup)
            
            if not use_lhs:
                for j in range(E):
                    field = sample_from_eof_basis(nao_bases, nao_phase, l, device, H, W)
                    nao_noise[b_idx*E + j, 0] = field
                    nao_noise[b_idx*E + j, 1] = field.clone()
            else:
                fields = sample_batch_lhs(nao_bases, nao_phase, l, device, H, W, E)
                nao_noise[b_idx*E:(b_idx+1)*E, 0] = fields
                nao_noise[b_idx*E:(b_idx+1)*E, 1] = fields.clone()
            
    # ── ENSO EOFs ──
    enso_noise = torch.randn((vB * E, 2, H, W), device=device)
    if enso_bases is not None and oni_lookup is not None:
        months = batch['month']
        leads = batch['lead_idx']
        for b_idx in range(vB):
            m = int(months[b_idx])
            l = int(leads[b_idx])
            enso_state = get_enso_state(m, year, oni_lookup)
            
            if not use_lhs:
                for j in range(E):
                    field = sample_from_eof_basis(enso_bases, enso_state, l, device, H, W)
                    enso_noise[b_idx*E + j, 0] = field
                    enso_noise[b_idx*E + j, 1] = field.clone()
            else:
                fields = sample_batch_lhs(enso_bases, enso_state, l, device, H, W, E)
                enso_noise[b_idx*E:(b_idx+1)*E, 0] = fields
                enso_noise[b_idx*E:(b_idx+1)*E, 1] = fields.clone()
            
    # ── Compute Dynamic Amplitudes ──
    month_val = int(batch['month'][0])
    init_date = datetime.date(year, month_val, 15)
    
    # MJO amp
    mjo_amp = 1.0
    date_str = init_date.strftime('%Y-%m-%d')
    if mjo_df is not None and date_str in mjo_df.index:
        row = mjo_df.loc[date_str]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        r1, r2 = row.get('RMM1_lagged', 0.0), row.get('RMM2_lagged', 0.0)
        if not (pd.isna(r1) or pd.isna(r2)):
            mjo_amp = float(np.sqrt(r1**2 + r2**2))
            
    # NAO amp
    nao_amp = 0.5
    if nao_lookup is not None:
        nao_amp = abs(get_nao_value(init_date, nao_lookup))
        
    # ENSO amp
    enso_amp = 0.5
    if oni_lookup is not None:
        enso_amp = abs(get_enso_value(month_val, year, oni_lookup))
        
    # Cap / Floor
    mjo_amp = max(min(mjo_amp, 3.0), 0.1)
    nao_amp = max(min(nao_amp, 2.5), 0.1)
    enso_amp = max(min(enso_amp, 2.5), 0.1)
    
    total = mjo_amp + nao_amp + enso_amp
    w_mjo = mjo_amp / total
    w_nao = nao_amp / total
    w_enso = enso_amp / total
    
    blend = 0.90 * (w_mjo * mjo_noise + w_nao * nao_noise + w_enso * enso_noise) + 0.10 * pure_noise
    std = blend.std(dim=(2, 3), keepdim=True)
    return blend / (std + 1e-6)

def orthogonalize_noise_batch(noise, vB, E):
    """
    Gram-Schmidt orthogonalization across the ensemble dimension (E) for each sample in the batch (vB).
    noise: shape [vB * E, 1, H, W]
    Returns: shape [vB * E, 1, H, W] orthogonalized and standardized noise.
    """
    noise_reshaped = noise.view(vB, E, -1) # [vB, E, H*W]
    out = torch.zeros_like(noise_reshaped)
    
    for b in range(vB):
        for i in range(E):
            v = noise_reshaped[b, i].clone()
            for j in range(i):
                u = out[b, j]
                proj = torch.sum(v * u) / (torch.sum(u * u) + 1e-8)
                v = v - proj * u
            std = v.std()
            if std > 1e-6:
                v = v / std
            out[b, i] = v
            
    return out.view(noise.shape)
