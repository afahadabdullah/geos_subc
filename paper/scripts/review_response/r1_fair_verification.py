#!/usr/bin/env python3
"""Review response M1/M2/M3/M5/M7: verification suite (lean by default).

Default components (cheap, one pass over the Zarrs after a light obs pass):

  matched : headline scores using fixed ML subset size and native lagged FIMr1p1 (M2)
            - FIMr1p1 reference: 8-member lagged ensemble (4 members from each
              of the two initializations verifying the same target week)
            - ML forecast: 16-member subsets of the 90-member generated
              ensemble, averaged over repeated draws
  clim    : leave-one-year-out (LOYO) weekly climatological ensemble reference
            and CRPSS versus climatology for every system (M1)
  debias  : LOYO lead/month mean-debiased FIMr1p1 baseline (M1)
  acc     : anomaly correlation of ensemble means vs LOYO climatology (M5)
  boot    : moving-block bootstrap CIs over start dates (M3)
  rank    : rank-histogram counts + spread/RMSE by lead (M7)
  qm      : frozen empirical quantile-mapped FIMr1p1 baseline; requires
            --qm_dir from qm_fim_baseline.py

Opt-in extras (slower): fair (Ferro fair CRPS), emos (T2M EMOS-lite baseline).

Usage:
  python paper/scripts/review_response/r1_fair_verification.py \
      --forecast_dir dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50 \
      --years 2021,2022,2023

Useful flags:
  --components matched,clim,debias,acc,boot,rank,qm,fair,emos
  --model_members 16     # ML subset size; FIMr1p1 lagged ensemble remains native
  --eval_mask land --land_mask_file ml_model/land_ocean_mask_v6.pt
  --max_inits 6          # smoke test
  --threshold_file <nc>  # long-term observed q95 thresholds (else LOYO monthly)
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARIABLES = {
    "pr": {"model": "model_pr", "geos": "geos_pr", "obs": "obs_pr", "min_threshold": 5.0},
    "t2m": {"model": "model_t2m", "geos": "geos_t2m", "obs": "obs_t2m", "min_threshold": None},
}
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast_dir",
                   default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50")
    p.add_argument(
        "--qm_dir",
        default=None,
        help="Directory containing <year>.zarr stores with qm_pr/qm_t2m.",
    )
    p.add_argument("--years", default="2021,2022,2023")
    p.add_argument("--out_dir",
                   default="ml_output_flow_finalv1_global_noisectx_t2mres/r1_fair_verification")
    p.add_argument("--variables", default="pr,t2m")
    p.add_argument("--components", default="matched,clim,debias,acc,boot,rank",
                   help="Comma list from: matched,clim,debias,acc,boot,rank,qm,fair,emos")
    p.add_argument("--model_members", "--match_members", dest="model_members",
                   type=int, default=16,
                   help="Number of generated ML members to sample; FIMr1p1 uses its native lagged ensemble.")
    p.add_argument("--eval_mask", choices=("all", "land"), default="all")
    p.add_argument("--land_mask_file", default="ml_model/land_ocean_mask_v6.pt")
    p.add_argument("--clim_window_weeks", type=int, default=2)
    p.add_argument("--subsample_repeats", type=int, default=20)
    p.add_argument("--extreme_quantile", type=float, default=0.95)
    p.add_argument("--threshold_file", default=None)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--block_len", type=int, default=4)
    p.add_argument("--rank_stride", type=int, default=2)
    p.add_argument("--max_inits", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric primitives (member axis first)
# ---------------------------------------------------------------------------

def crps_terms(ens: np.ndarray, obs: np.ndarray):
    ens = ens.astype(np.float64, copy=False)
    obs = obs.astype(np.float64, copy=False)
    m = ens.shape[0]
    valid = np.isfinite(obs) & np.all(np.isfinite(ens), axis=0)
    term1 = np.nanmean(np.abs(ens - obs[None]), axis=0)
    srt = np.sort(ens, axis=0)
    coeff = (2.0 * np.arange(1, m + 1, dtype=np.float64) - m - 1.0)
    gini = np.tensordot(coeff, srt, axes=(0, 0))
    return term1, gini, valid


def crps_standard(ens, obs):
    m = ens.shape[0]
    term1, gini, valid = crps_terms(ens, obs)
    return np.where(valid, term1 - gini / (m * m), np.nan), valid


def crps_fair(ens, obs):
    m = ens.shape[0]
    term1, gini, valid = crps_terms(ens, obs)
    if m < 2:
        return np.where(valid, term1, np.nan), valid
    return np.where(valid, term1 - gini / (m * (m - 1)), np.nan), valid


def gaussian_crps(mu, sigma, obs):
    z = (obs - mu) / max(sigma, 1e-9)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def wsum(field, w):
    ok = np.isfinite(field) & (w > 0)
    if not ok.any():
        return 0.0, 0.0
    return float(np.sum(field[ok] * w[ok])), float(np.sum(w[ok]))


def area_weights(lats, lons, mask_mode, land_mask_file):
    w = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, len(lons)))
    w = np.clip(w, 0.0, None)
    if mask_mode == "land":
        import torch
        blob = torch.load(land_mask_file, map_location="cpu", weights_only=False)
        key = "is_land" if (hasattr(blob, "keys") and "is_land" in blob) else "land_mask"
        land = np.asarray(blob[key] if hasattr(blob, "keys") else blob, dtype=float)
        w = w * (land > 0.5).astype(float)
    return w


def doy_distance(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    diff = abs(d1.dayofyear - d2.dayofyear)
    return min(diff, 365 - diff)


def load_threshold_file(path, variables):
    ds = xr.open_dataset(path)
    out = {}
    for var in variables:
        cand = [v for v in ds.data_vars if var in v.lower() and "thresh" in v.lower()] or \
               [v for v in ds.data_vars if var in v.lower()]
        if cand:
            out[var] = ds[cand[0]].load()
    return out


def threshold_for(var, valid, loaded, loyo_thresholds, year):
    if loaded is not None and var in loaded:
        da = loaded[var]
        if "month" in da.dims:
            return np.asarray(da.sel(month=valid.month).values, dtype=np.float64)
        return np.asarray(da.values, dtype=np.float64)
    return loyo_thresholds.get((var, year, valid.month))


def lagged_partner(ii: int, lead: int, n_init: int, leads: list[int]):
    """Return (init_idx, lead_value) of the other initialization verifying the
    same target week, or None. Prefers the older initialization (proper lagged
    ensemble); falls back to the adjacent newer one at the longest lead."""
    if (lead + 1) in leads and ii - 1 >= 0:
        return ii - 1, lead + 1
    if (lead - 1) in leads and ii + 1 < n_init:
        return ii + 1, lead - 1
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    years = [int(y) for y in args.years.split(",") if y.strip()]
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    comps = {c.strip() for c in args.components.split(",") if c.strip()}
    if args.qm_dir:
        comps.add("qm")
    if "qm" in comps and not args.qm_dir:
        raise ValueError("The qm component requires --qm_dir.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "matched" in comps:
        print(
            f"Matched protocol: ML samples up to {args.model_members} generated members; "
            "FIMr1p1 uses native lagged ensemble members."
        )

    need_pass1 = bool(comps & {"clim", "debias", "acc", "emos"})

    obs_by_valid = {v: {} for v in variables}
    bias_sums, bias_counts = {}, defaultdict(int)
    emos = defaultdict(lambda: np.zeros(6))
    lats = lons = w2d = None

    if need_pass1:
        print("Pass 1: collecting observations / bias fields / EMOS moments ...")
        for year in years:
            ds = xr.open_zarr(Path(args.forecast_dir) / f"{year}.zarr",
                              consolidated=False, chunks=None)
            try:
                if lats is None:
                    lats = np.asarray(ds["lat"].values, dtype=float)
                    lons = np.asarray(ds["lon"].values, dtype=float)
                    w2d = area_weights(lats, lons, args.eval_mask, args.land_mask_file)
                inits = pd.to_datetime(ds["init"].values).normalize()
                leads = [int(l) for l in ds["lead"].values]
                n_init = len(inits) if args.max_inits is None else min(len(inits), args.max_inits)
                for ii in range(n_init):
                    for li, lead in enumerate(leads):
                        valid = pd.Timestamp(inits[ii] + pd.Timedelta(days=7 * lead))
                        for var in variables:
                            spec = VARIABLES[var]
                            obs = ds[spec["obs"]].isel(init=ii, lead=li).values.astype(np.float32)
                            obs_by_valid[var].setdefault(valid, obs)
                            if not (comps & {"debias", "emos"}):
                                continue
                            gmean = np.nanmean(ds[spec["geos"]].isel(init=ii, lead=li).values,
                                               axis=0).astype(np.float32)
                            if "debias" in comps:
                                key = (var, year, valid.month, lead)
                                diff = obs.astype(np.float64) - gmean
                                if key not in bias_sums:
                                    bias_sums[key] = np.zeros_like(diff)
                                np.add(bias_sums[key], np.where(np.isfinite(diff), diff, 0.0),
                                       out=bias_sums[key])
                                bias_counts[key] += 1
                            if "emos" in comps and var == "t2m":
                                season = SEASON_OF_MONTH[valid.month]
                                ok = np.isfinite(obs) & np.isfinite(gmean) & (w2d > 0)
                                g, o, ww = gmean[ok], obs[ok].astype(np.float64), w2d[ok]
                                emos[(year, lead, season)] += np.array([
                                    ww.sum(), (ww * g).sum(), (ww * o).sum(),
                                    (ww * g * o).sum(), (ww * g * g).sum(), (ww * o * o).sum()])
            finally:
                ds.close()

    def emos_coeffs(target_year, lead, season):
        tot = np.zeros(6)
        for (yy, ll, ss), acc in emos.items():
            if yy != target_year and ll == lead and ss == season:
                tot += acc
        n, sg, so, sgo, sgg, soo = tot
        if n <= 1:
            return None
        gbar, obar = sg / n, so / n
        var_g = sgg / n - gbar ** 2
        if var_g <= 1e-9:
            return None
        b = (sgo / n - gbar * obar) / var_g
        a = obar - b * gbar
        resid_var = max(soo / n - 2 * a * obar - 2 * b * sgo / n + a * a
                        + 2 * a * b * gbar + b * b * sgg / n, 1e-6)
        return a, b, math.sqrt(resid_var)

    loaded_thr = load_threshold_file(args.threshold_file, variables) if args.threshold_file else None
    loyo_thresholds = {}
    if loaded_thr is None and obs_by_valid[variables[0]]:
        print("Building LOYO monthly extreme thresholds ...")
        for var in variables:
            for year in years:
                for month in range(1, 13):
                    pool = [f for d, f in obs_by_valid[var].items()
                            if d.month == month and d.year != year]
                    if len(pool) < 4:
                        pool = [f for d, f in obs_by_valid[var].items() if d.month == month]
                    if not pool:
                        continue
                    thr = np.nanquantile(np.stack(pool).astype(np.float64),
                                         args.extreme_quantile, axis=0)
                    floor = VARIABLES[var]["min_threshold"]
                    if floor is not None:
                        thr = np.maximum(thr, floor)
                    loyo_thresholds[(var, year, month)] = thr

    def clim_members(var, valid):
        window = 7 * args.clim_window_weeks
        return [f for d, f in obs_by_valid[var].items()
                if d.year != valid.year and doy_distance(d, valid) <= window]

    # ---------------- Pass 2: scoring --------------------------------------
    print("Pass 2: scoring ...")
    rows = []
    rank_counts = {}
    stride = max(1, args.rank_stride)

    for year in years:
        ds = xr.open_zarr(Path(args.forecast_dir) / f"{year}.zarr",
                          consolidated=False, chunks=None)
        qm_ds = None
        if "qm" in comps:
            qm_path = Path(args.qm_dir) / f"{year}.zarr"
            if not qm_path.exists():
                ds.close()
                raise FileNotFoundError(f"QM forecast archive not found: {qm_path}")
            qm_ds = xr.open_zarr(qm_path, consolidated=False, chunks=None)
            source_inits = pd.to_datetime(ds["init"].values).normalize()
            qm_inits = pd.to_datetime(qm_ds["init"].values).normalize()
            if not np.array_equal(source_inits.values, qm_inits.values):
                ds.close()
                qm_ds.close()
                raise ValueError(f"QM/source initialization dates differ for {year}.")
            if not np.array_equal(ds["lead"].values, qm_ds["lead"].values):
                ds.close()
                qm_ds.close()
                raise ValueError(f"QM/source lead coordinates differ for {year}.")
        try:
            if lats is None:
                lats = np.asarray(ds["lat"].values, dtype=float)
                lons = np.asarray(ds["lon"].values, dtype=float)
                w2d = area_weights(lats, lons, args.eval_mask, args.land_mask_file)
            inits = pd.to_datetime(ds["init"].values).normalize()
            leads = [int(l) for l in ds["lead"].values]
            n_init = len(inits) if args.max_inits is None else min(len(inits), args.max_inits)
            for ii in range(n_init):
                for li, lead in enumerate(leads):
                    valid = pd.Timestamp(inits[ii] + pd.Timedelta(days=7 * lead))
                    season = SEASON_OF_MONTH[valid.month]
                    for var in variables:
                        spec = VARIABLES[var]
                        obs = ds[spec["obs"]].isel(init=ii, lead=li).values.astype(np.float64)
                        model = ds[spec["model"]].isel(init=ii, lead=li).values
                        geos = ds[spec["geos"]].isel(init=ii, lead=li).values
                        qm = None
                        if qm_ds is not None:
                            qm = qm_ds[f"qm_{var}"].isel(init=ii, lead=li).values

                        row = {"variable": var, "year": year, "lead": lead,
                               "init_time": inits[ii], "season": season,
                               "n_members_model": int(model.shape[0]),
                               "n_members_geos": int(geos.shape[0])}

                        fields = {}
                        c_model, _ = crps_standard(model, obs)
                        c_geos, _ = crps_standard(geos, obs)
                        fields["crps_model"] = c_model
                        fields["crps_geos"] = c_geos

                        # Repeated fixed-K flow subsets. With
                        # --model_members 8 these fields provide the requested
                        # flow-8 versus native raw-4/QM-4 comparison. The
                        # separate model_qm_m fields below remain the strict
                        # equal-member flow-4 versus QM-4 control.
                        if "matched" in comps or qm is not None:
                            k = min(args.model_members, model.shape[0])
                            row["n_members_model_subsample"] = int(k)
                            acc_c = np.zeros_like(c_model)
                            acc_m = np.zeros_like(c_model)
                            acc_s = np.zeros_like(c_model)
                            for _ in range(args.subsample_repeats):
                                idx = rng.choice(
                                    model.shape[0], size=k, replace=False
                                )
                                subset = model[idx]
                                subset_crps, _ = crps_standard(subset, obs)
                                acc_c = np.nansum([acc_c, subset_crps], axis=0)
                                subset_float = subset.astype(np.float64)
                                subset_mean = np.nanmean(subset_float, axis=0)
                                acc_m = np.nansum(
                                    [acc_m, (subset_mean - obs) ** 2], axis=0
                                )
                                acc_s = np.nansum(
                                    [acc_s, np.nanstd(subset_float, axis=0)],
                                    axis=0,
                                )
                            fields["crps_model_m"] = (
                                acc_c / args.subsample_repeats
                            )
                            fields["mse_model_m"] = (
                                acc_m / args.subsample_repeats
                            )
                            fields["spread_model_m"] = (
                                acc_s / args.subsample_repeats
                            )

                        if qm is not None:
                            fields["crps_qm"], _ = crps_standard(qm, obs)
                            qm_mean = np.nanmean(qm.astype(np.float64), axis=0)
                            fields["mse_qm"] = (qm_mean - obs) ** 2
                            fields["spread_qm"] = np.nanstd(
                                qm.astype(np.float64), axis=0
                            )
                            row["n_members_qm"] = int(qm.shape[0])

                            # This comparison is always ensemble-size matched,
                            # independent of the lagged-reference configuration.
                            qm_k = min(qm.shape[0], model.shape[0])
                            row["n_members_model_qm_subsample"] = int(qm_k)
                            qm_acc_crps = np.zeros_like(c_model)
                            qm_acc_mse = np.zeros_like(c_model)
                            for _ in range(args.subsample_repeats):
                                idx = rng.choice(
                                    model.shape[0], size=qm_k, replace=False
                                )
                                subset_crps, _ = crps_standard(model[idx], obs)
                                qm_acc_crps = np.nansum(
                                    [qm_acc_crps, subset_crps], axis=0
                                )
                                subset_mean = np.nanmean(
                                    model[idx].astype(np.float64), axis=0
                                )
                                qm_acc_mse = np.nansum(
                                    [qm_acc_mse, (subset_mean - obs) ** 2], axis=0
                                )
                            fields["crps_model_qm_m"] = (
                                qm_acc_crps / args.subsample_repeats
                            )
                            fields["mse_model_qm_m"] = (
                                qm_acc_mse / args.subsample_repeats
                            )

                        # matched-ensemble protocol (M2)
                        if "matched" in comps:
                            partner = lagged_partner(ii, lead, n_init, leads)
                            if partner is not None:
                                pi, plead = partner
                                geos2 = ds[spec["geos"]].isel(
                                    init=pi, lead=leads.index(plead)).values
                                geos_lag = np.concatenate([geos, geos2], axis=0)
                                row["n_members_geos_lag"] = int(geos_lag.shape[0])
                                fields["crps_geos_lag"], _ = crps_standard(geos_lag, obs)
                                glmean = np.nanmean(geos_lag.astype(np.float64), axis=0)
                                fields["mse_geos_lag"] = (glmean - obs) ** 2

                        if "fair" in comps:
                            fields["faircrps_model"], _ = crps_fair(model, obs)
                            fields["faircrps_geos"], _ = crps_fair(geos, obs)

                        clim_mean = None
                        if "clim" in comps:
                            clim = clim_members(var, valid)
                            if len(clim) >= 4:
                                clim_ens = np.stack(clim)
                                fields["crps_clim"], _ = crps_standard(clim_ens, obs)
                                clim_mean = np.nanmean(clim_ens.astype(np.float64), axis=0)

                        if "debias" in comps:
                            tot, cnt = None, 0
                            for yy in years:
                                if yy == year:
                                    continue
                                key = (var, yy, valid.month, lead)
                                if key in bias_sums and bias_counts[key] > 0:
                                    tot = bias_sums[key] if tot is None else tot + bias_sums[key]
                                    cnt += bias_counts[key]
                            if tot is not None and cnt > 0:
                                geos_db = geos.astype(np.float64) + (tot / cnt)[None]
                                if var == "pr":
                                    geos_db = np.clip(geos_db, 0.0, None)
                                fields["crps_geos_debias"], _ = crps_standard(geos_db, obs)

                        if "emos" in comps and var == "t2m":
                            coeffs = emos_coeffs(year, lead, season)
                            if coeffs is not None:
                                a, b, sig = coeffs
                                gmean = np.nanmean(geos.astype(np.float64), axis=0)
                                ce = gaussian_crps(a + b * gmean, sig, obs)
                                fields["crps_emos"] = np.where(
                                    np.isfinite(obs) & np.isfinite(gmean), ce, np.nan)

                        mmean = np.nanmean(model.astype(np.float64), axis=0)
                        gmean = np.nanmean(geos.astype(np.float64), axis=0)
                        fields["mse_model"] = (mmean - obs) ** 2
                        fields["mse_geos"] = (gmean - obs) ** 2
                        fields["spread_model"] = np.nanstd(model.astype(np.float64), axis=0)

                        if "acc" in comps and clim_mean is not None:
                            a_o = obs - clim_mean
                            a_m = mmean - clim_mean
                            a_g = gmean - clim_mean
                            ok = (np.isfinite(a_o) & np.isfinite(a_m)
                                  & np.isfinite(a_g) & (w2d > 0))
                            ww = w2d[ok]
                            den_o = np.sqrt(np.sum(ww * a_o[ok] ** 2))
                            row["acc_model"] = float(np.sum(ww * a_m[ok] * a_o[ok]) / max(
                                np.sqrt(np.sum(ww * a_m[ok] ** 2)) * den_o, 1e-9))
                            row["acc_geos"] = float(np.sum(ww * a_g[ok] * a_o[ok]) / max(
                                np.sqrt(np.sum(ww * a_g[ok] ** 2)) * den_o, 1e-9))
                            if qm is not None:
                                a_q = qm_mean - clim_mean
                                ok_q = (
                                    np.isfinite(a_o)
                                    & np.isfinite(a_q)
                                    & (w2d > 0)
                                )
                                ww_q = w2d[ok_q]
                                den_o_q = np.sqrt(np.sum(ww_q * a_o[ok_q] ** 2))
                                row["acc_qm"] = float(
                                    np.sum(ww_q * a_q[ok_q] * a_o[ok_q])
                                    / max(
                                        np.sqrt(np.sum(ww_q * a_q[ok_q] ** 2))
                                        * den_o_q,
                                        1e-9,
                                    )
                                )

                        for name, field in fields.items():
                            s, wt = wsum(field, w2d)
                            row[f"{name}_wsum"] = s
                            row[f"{name}_w"] = wt

                        thr = threshold_for(var, valid, loaded_thr, loyo_thresholds, year)
                        if thr is not None:
                            w_ext = np.where((obs >= thr) & np.isfinite(obs), w2d, 0.0)
                            for name in ("crps_model", "crps_geos", "crps_geos_lag",
                                         "crps_model_m", "crps_clim", "crps_qm",
                                         "crps_model_qm_m"):
                                if name in fields:
                                    s, wt = wsum(fields[name], w_ext)
                                    row[f"ext_{name}_wsum"] = s
                                    row[f"ext_{name}_w"] = wt

                        if "rank" in comps:
                            sub_obs = obs[::stride, ::stride]
                            sub_ens = model[:, ::stride, ::stride].astype(np.float64)
                            sub_w = w2d[::stride, ::stride]
                            ok = (np.isfinite(sub_obs)
                                  & np.all(np.isfinite(sub_ens), axis=0) & (sub_w > 0))
                            if ok.any():
                                below = np.sum(sub_ens[:, ok] < sub_obs[ok][None], axis=0)
                                ties = np.sum(sub_ens[:, ok] == sub_obs[ok][None], axis=0)
                                ranks = below + (rng.random(below.shape) * (ties + 1)).astype(int)
                                key = (var, lead)
                                if key not in rank_counts:
                                    rank_counts[key] = np.zeros(model.shape[0] + 1)
                                np.add.at(rank_counts[key], ranks, sub_w[ok])

                        rows.append(row)
                print(f"  {year} init {ii + 1}/{n_init} done", flush=True)
        finally:
            ds.close()
            if qm_ds is not None:
                qm_ds.close()

    per_init = pd.DataFrame(rows)
    per_init.to_csv(out_dir / "r1_per_init_metrics.csv", index=False)

    # ---------------- Aggregation + bootstrap ------------------------------
    def agg_mean(df, name):
        if f"{name}_wsum" not in df:
            return float("nan")
        return df[f"{name}_wsum"].sum() / max(df[f"{name}_w"].sum(), 1e-9)

    def agg_ratio(df, num, den):
        a, b = agg_mean(df, num), agg_mean(df, den)
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-12:
            return float("nan")
        return 100.0 * (1.0 - a / b)

    def agg_rmse_ratio(df, num, den):
        a, b = agg_mean(df, num), agg_mean(df, den)
        if not (np.isfinite(a) and np.isfinite(b)) or b <= 1e-12:
            return float("nan")
        return 100.0 * (1.0 - math.sqrt(max(a, 0.0)) / math.sqrt(b))

    MEANS = ["crps_model", "crps_geos", "crps_model_m", "crps_geos_lag",
             "crps_qm", "crps_model_qm_m",
             "crps_clim", "crps_geos_debias", "crps_emos",
             "faircrps_model", "faircrps_geos"]
    SKILLS = [
        ("skill_matched", "crps_model_m", "crps_geos_lag"),
        ("rmse_skill_matched", "mse_model_m", "mse_geos_lag"),
        ("skill_full90_vs_raw4", "crps_model", "crps_geos"),
        ("skill_fair", "faircrps_model", "faircrps_geos"),
        ("crpss_clim_model_m", "crps_model_m", "crps_clim"),
        ("crpss_clim_geos_lag", "crps_geos_lag", "crps_clim"),
        ("crpss_clim_geos_debias", "crps_geos_debias", "crps_clim"),
        ("crpss_clim_emos", "crps_emos", "crps_clim"),
        ("skill_vs_debias", "crps_model_m", "crps_geos_debias"),
        ("skill_vs_emos", "crps_model_m", "crps_emos"),
        ("skill_qm_vs_raw", "crps_qm", "crps_geos"),
        ("rmse_skill_qm_vs_raw", "mse_qm", "mse_geos"),
        ("skill_model_k_vs_raw", "crps_model_m", "crps_geos"),
        ("rmse_skill_model_k_vs_raw", "mse_model_m", "mse_geos"),
        ("skill_model_k_vs_qm", "crps_model_m", "crps_qm"),
        ("rmse_skill_model_k_vs_qm", "mse_model_m", "mse_qm"),
        ("skill_model_vs_qm", "crps_model_qm_m", "crps_qm"),
        ("rmse_skill_model_vs_qm", "mse_model_qm_m", "mse_qm"),
        ("crpss_clim_qm", "crps_qm", "crps_clim"),
        ("ext_skill_matched", "ext_crps_model_m", "ext_crps_geos_lag"),
        ("ext_crpss_clim_model", "ext_crps_model_m", "ext_crps_clim"),
        ("ext_skill_qm_vs_raw", "ext_crps_qm", "ext_crps_geos"),
        ("ext_skill_model_k_vs_raw", "ext_crps_model_m", "ext_crps_geos"),
        ("ext_skill_model_k_vs_qm", "ext_crps_model_m", "ext_crps_qm"),
        ("ext_skill_model_vs_qm", "ext_crps_model_qm_m", "ext_crps_qm"),
    ]

    def block_bootstrap_ci(df, num, den, *, rmse=False):
        inits_sorted = sorted(df["init_time"].unique())
        n = len(inits_sorted)
        if n < 8:
            return float("nan"), float("nan")
        L = max(1, args.block_len)
        groups = {t: g for t, g in df.groupby("init_time")}
        stats = []
        for _ in range(args.n_boot):
            picked = []
            while len(picked) < n:
                s = rng.integers(0, n)
                picked.extend(inits_sorted[s: s + L])
            sample = pd.concat([groups[t] for t in picked[:n]], ignore_index=True)
            ratio_fn = agg_rmse_ratio if rmse else agg_ratio
            stats.append(ratio_fn(sample, num, den))
        stats = np.asarray(stats)
        stats = stats[np.isfinite(stats)]
        if stats.size < 50:
            return float("nan"), float("nan")
        return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

    agg_rows = []
    year_groups = [("pooled", per_init)] + [(str(y), per_init[per_init["year"].eq(y)])
                                            for y in years]
    for year_label, ydf in year_groups:
        for var in variables:
            vdf = ydf[ydf["variable"].eq(var)]
            lead_groups = [("all", vdf)] + [(str(l), vdf[vdf["lead"].eq(l)])
                                            for l in sorted(vdf["lead"].unique())]
            for lead_label, ldf in lead_groups:
                if ldf.empty:
                    continue
                row = {"year": year_label, "variable": var, "lead": lead_label,
                       "n_init_rows": len(ldf)}
                for name in MEANS:
                    row[name] = agg_mean(ldf, name)
                for sname, num, den in SKILLS:
                    ratio_fn = agg_rmse_ratio if sname.startswith("rmse_") else agg_ratio
                    row[sname] = ratio_fn(ldf, num, den)
                rmse_m = math.sqrt(max(agg_mean(ldf, "mse_model"), 0.0))
                row["rmse_model_full"] = rmse_m
                row["spread_rmse_model"] = agg_mean(ldf, "spread_model") / max(rmse_m, 1e-9)
                if "mse_model_m_wsum" in ldf:
                    rmse_model_k = math.sqrt(
                        max(agg_mean(ldf, "mse_model_m"), 0.0)
                    )
                    row["rmse_model_k"] = rmse_model_k
                    row["spread_rmse_model_k"] = (
                        agg_mean(ldf, "spread_model_m")
                        / max(rmse_model_k, 1e-9)
                    )
                if "mse_qm_wsum" in ldf:
                    rmse_qm = math.sqrt(max(agg_mean(ldf, "mse_qm"), 0.0))
                    row["rmse_qm"] = rmse_qm
                    row["spread_rmse_qm"] = (
                        agg_mean(ldf, "spread_qm") / max(rmse_qm, 1e-9)
                    )
                if "acc_model" in ldf:
                    row["acc_model"] = float(ldf["acc_model"].mean())
                    row["acc_geos"] = float(ldf["acc_geos"].mean())
                    if "acc_qm" in ldf:
                        row["acc_qm"] = float(ldf["acc_qm"].mean())
                if "boot" in comps and year_label == "pooled" and lead_label == "all":
                    bootstrap_names = {
                        "skill_matched",
                        "rmse_skill_matched",
                        "crpss_clim_model_m",
                        "crpss_clim_geos_lag",
                        "skill_qm_vs_raw",
                        "rmse_skill_qm_vs_raw",
                        "skill_model_k_vs_raw",
                        "rmse_skill_model_k_vs_raw",
                        "skill_model_k_vs_qm",
                        "rmse_skill_model_k_vs_qm",
                        "skill_model_vs_qm",
                        "rmse_skill_model_vs_qm",
                        "crpss_clim_qm",
                        "ext_skill_matched",
                        "ext_skill_qm_vs_raw",
                        "ext_skill_model_k_vs_raw",
                        "ext_skill_model_k_vs_qm",
                        "ext_skill_model_vs_qm",
                    }
                    for sname, num, den in SKILLS:
                        if sname not in bootstrap_names:
                            continue
                        lo, hi = block_bootstrap_ci(
                            ldf,
                            num,
                            den,
                            rmse=sname.startswith("rmse_"),
                        )
                        row[f"{sname}_ci_lo"] = lo
                        row[f"{sname}_ci_hi"] = hi
                for member_column in (
                    "n_members_model_subsample",
                    "n_members_geos",
                    "n_members_qm",
                    "n_members_model_qm_subsample",
                ):
                    if member_column in ldf:
                        counts = ldf[member_column].dropna().unique()
                        if len(counts) == 1:
                            row[member_column] = int(counts[0])
                agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "r1_aggregate_metrics.csv", index=False)

    if rank_counts:
        rank_rows = []
        for (var, lead), counts in sorted(rank_counts.items()):
            total = counts.sum()
            for b, c in enumerate(counts):
                rank_rows.append({"variable": var, "lead": lead, "rank_bin": b,
                                  "weight": float(c),
                                  "frequency": float(c / max(total, 1e-9))})
        pd.DataFrame(rank_rows).to_csv(out_dir / "r1_rank_histogram.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print("\n================= R1 SUMMARY (paste this back) =================")
    show = ["year", "variable", "lead", "skill_matched", "rmse_skill_matched",
            "skill_full90_vs_raw4", "skill_fair",
            "crpss_clim_model_m", "crpss_clim_geos_lag", "crpss_clim_geos_debias",
            "crpss_clim_emos", "skill_vs_debias", "skill_vs_emos",
            "skill_qm_vs_raw", "rmse_skill_qm_vs_raw",
            "n_members_model_subsample", "n_members_geos", "n_members_qm",
            "skill_model_k_vs_raw", "rmse_skill_model_k_vs_raw",
            "skill_model_k_vs_qm", "rmse_skill_model_k_vs_qm",
            "skill_model_vs_qm", "rmse_skill_model_vs_qm",
            "crpss_clim_qm", "ext_skill_matched", "ext_crpss_clim_model",
            "ext_skill_qm_vs_raw", "ext_skill_model_k_vs_raw",
            "ext_skill_model_k_vs_qm", "ext_skill_model_vs_qm",
            "acc_model", "acc_geos", "acc_qm",
            "spread_rmse_model", "spread_rmse_model_k", "spread_rmse_qm"]
    avail = [c for c in show if c in agg.columns]
    print(agg[agg["lead"].eq("all")][avail].round(3).to_string(index=False))
    per_lead = agg[(agg["year"].eq("pooled")) & (~agg["lead"].eq("all"))]
    print("\n--- pooled, by lead ---")
    print(per_lead[avail].round(3).to_string(index=False))
    ci_cols = ["variable"] + [c for c in agg.columns if "_ci_" in c]
    if len(ci_cols) > 1:
        print("\n--- 95% block-bootstrap CIs (pooled, all leads) ---")
        print(agg[(agg["year"].eq("pooled")) & (agg["lead"].eq("all"))][ci_cols]
              .round(3).to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
