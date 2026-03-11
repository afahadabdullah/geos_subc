"""
check_variable.py
=================
Audits the availability of all weekly Zarr files required by the training
pipeline (dataset_flow.py / S2SHybridDataset) for years 1999-2025.

Variables checked (mirrors dataset_flow.py file loading logic):
    - geos_subc        : GEOS S2S Forecast    [CORE - required]
    - gpcp_weekly      : GPCP Precipitation   [CORE - required]
    - sst_weekly       : Sea Surface Temp
    - sss_weekly       : Sea Surface Salinity
    - soilw_weekly     : Soil Moisture
    - ivt_weekly       : Integrated Vapor Transport
    - mjowave_weekly   : Spatial MJO Wave Envelope
    - z500_u250_weekly : Z500 & U250 (ERA5)

Usage:
    python dataprocess/check_variable.py
    python dataprocess/check_variable.py --data_root /path/to/dataprocess
    python dataprocess/check_variable.py --data_root /path/to/dataprocess --open  # Also open Zarrs to verify dims
"""
import os
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Configuration – mirrors dataset_flow.py file naming conventions
# ─────────────────────────────────────────────────────────────────────────────
YEAR_RANGE = range(1999, 2026)  # 1999 to 2025 inclusive

# (display_name, file_template, is_core, required_vars)
VARIABLES = [
    ("GEOS Forecast",           "geos_subc_{year}.zarr",        True,  ["t2m", "z850"]),
    ("GPCP Precipitation",      "gpcp_weekly_{year}.zarr",      True,  []),
    ("Sea Surface Temp (SST)",  "sst_weekly_{year}.zarr",       False, []),
    ("Sea Surface Sal (SSS)",   "sss_weekly_{year}.zarr",       False, []),
    ("Soil Moisture (SoilW)",   "soilw_weekly_{year}.zarr",     False, []),
    ("IVT (ERA5)",              "ivt_weekly_{year}.zarr",       False, []),
    ("MJO Wave Envelope",       "mjowave_weekly_{year}.zarr",   False, []),
    ("Z500 & U250 (ERA5)",      "z500_u250_weekly_{year}.zarr", False, []),
    ("T2M (ERA5 Targets)",      "t2m_weekly_{year}.zarr",       False, []),
    ("SLP (ERA5 Targets)",      "slp_weekly_{year}.zarr",       False, []),
]

# Column widths for the table
COL_VAR   = 26
COL_YEAR  = 6
COL_YEARS = 27  # space for ~10 years in "Y Y Y" format

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def check_file(path: str, open_zarr: bool = False, required_vars: list = None):
    """
    Returns a status string: 'OK', 'MISSING', or 'EMPTY/ERR'.
    If open_zarr=True, also opens the Zarr store to verify readability,
    prints basic dimension info, and checks for required variables.
    """
    if not os.path.exists(path):
        return "MISSING"
    if open_zarr:
        try:
            import xarray as xr
            ds = xr.open_zarr(path, consolidated=False)
            dims = dict(ds.sizes)
            
            missing_vars = []
            if required_vars:
                ds_vars = [str(v).lower() for v in ds.variables]
                for req_v in required_vars:
                    if req_v.lower() not in ds_vars:
                        missing_vars.append(req_v)
            
            ds.close()
            # OK with dims embedded
            dim_str = " ".join(f"{k}={v}" for k, v in dims.items())
            
            if missing_vars:
                return f"ERR (Missing vars: {','.join(missing_vars)}) [{dim_str}]"
            return f"OK [{dim_str}]"
        except Exception as e:
            return f"ERR ({e})"
    return "OK"


def colored(text: str, color: str) -> str:
    """Simple ANSI coloring – degrades gracefully if terminal doesn't support it."""
    codes = {"green": "\033[92m", "red": "\033[91m",
             "yellow": "\033[93m", "reset": "\033[0m"}
    return f"{codes.get(color, '')}{text}{codes.get('reset', '')}"


# ─────────────────────────────────────────────────────────────────────────────
# Main audit
# ─────────────────────────────────────────────────────────────────────────────
def audit(data_root: str = "dataprocess", open_zarr: bool = False):
    print(f"\n{'='*80}")
    print(f"  Data Availability Audit  |  root: {os.path.abspath(data_root)}")
    print(f"  Years checked: {YEAR_RANGE.start} – {YEAR_RANGE.stop - 1}")
    print(f"{'='*80}\n")

    # ── Table header ──────────────────────────────────────────────────────────
    header = f"{'Variable':<{COL_VAR}}  {'#OK':>4}  {'Missing Years'}"
    print(header)
    print("-" * (COL_VAR + 4 + COL_YEARS + 4))

    summary: dict = {}   # variable_name -> {"ok": list, "missing": list}

    for display, template, is_core, required_vars in VARIABLES:
        ok_years, missing_years = [], []

        for year in YEAR_RANGE:
            path = os.path.join(data_root, template.format(year=year))
            status = check_file(path, open_zarr, required_vars=required_vars)
            if status.startswith("OK"):
                ok_years.append(year)
            else:
                missing_years.append(year)

        n_ok = len(ok_years)
        n_total = len(list(YEAR_RANGE))

        # Format missing years as compact ranges, e.g. 2017-2025
        missing_str = _format_year_list(missing_years)

        # Colour
        if n_ok == n_total:
            label = colored(f"{display:<{COL_VAR}}", "green")
            count = colored(f"{n_ok:>4}/{n_total}", "green")
        elif n_ok == 0:
            label = colored(f"{display:<{COL_VAR}}", "red")
            count = colored(f"{n_ok:>4}/{n_total}", "red")
        else:
            label = colored(f"{display:<{COL_VAR}}", "yellow")
            count = colored(f"{n_ok:>4}/{n_total}", "yellow")

        print(f"{label}  {count}  {missing_str}")
        summary[display] = {"ok": ok_years, "missing": missing_years, "core": is_core}

    # ── Year-by-year cross matrix ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Year-by-Year Availability Matrix  (C=core required, .=ok, X=missing)")
    print("─" * 80)

    var_labels = [v[0][:4] for v in VARIABLES]   # first 4 chars as column header
    header_row = f"  {'Year':<6}" + "".join(f"{lbl:<8}" for lbl in var_labels)
    print(header_row)
    print("─" * (8 + len(VARIABLES) * 8))

    for year in YEAR_RANGE:
        row = f"  {year:<6}"
        for display, template, is_core, _ in VARIABLES:
            path = os.path.join(data_root, template.format(year=year))
            exists = os.path.exists(path)
            marker = "." if exists else ("C" if is_core else "X")
            colour = "green" if exists else ("red" if is_core else "yellow")
            row += colored(f"{marker:<8}", colour)
        print(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  Summary")
    print("─" * 80)

    any_core_missing = False
    for display, info in summary.items():
        missing = info["missing"]
        is_core = info["core"]
        if missing:
            tag = "[CORE]" if is_core else "      "
            if is_core:
                any_core_missing = True
                print(colored(f"  {tag} {display}: {len(missing)} years missing → {_format_year_list(missing)}", "red"))
            else:
                print(colored(f"  {tag} {display}: {len(missing)} years missing → {_format_year_list(missing)}", "yellow"))
        else:
            print(colored(f"         {display}: Complete ✓", "green"))

    if any_core_missing:
        print(colored("\n  ⚠️  CORE files missing! Training will silently skip those years.", "red"))
    else:
        print(colored("\n  ✓ All CORE files present.", "green"))
    print(f"{'='*80}\n")


def _format_year_list(years: list) -> str:
    """Compact representation: consecutive ranges e.g. [2017-2019, 2023]."""
    if not years:
        return "—"
    ranges = []
    start = years[0]
    end = years[0]
    for y in years[1:]:
        if y == end + 1:
            end = y
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = y
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(ranges)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit weekly Zarr data availability for the S2S training pipeline."
    )
    parser.add_argument(
        "--data_root", type=str, default="dataprocess",
        help="Root directory that contains the weekly *.zarr files (default: dataprocess)"
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Actually open each Zarr store to verify readability and print dimension sizes (slower)."
    )
    args = parser.parse_args()
    audit(data_root=args.data_root, open_zarr=args.open)
