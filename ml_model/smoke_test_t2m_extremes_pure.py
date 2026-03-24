#!/usr/bin/env python3
"""
Smoke-test pure-noise T2M extreme-event forecasts in anomaly space.

Default assumption:
- use the pure-noise 2020-2021 forecast stores
- evaluate only 2021 init-year events so the forecast anomalies span 2021-2022
- use system-specific weekly climatology stores under dataprocess/clim_pure
"""

import sys

from smoke_test_t2m_extremes import main as smoke_main


DEFAULT_ARGS = [
    "--ml_dir", "dataprocess/gen_multiv1_pure_2020_2021",
    "--anomaly_mode", "system_store",
    "--ml_clim_path", "dataprocess/clim_pure/ml_weekly_ensmean_clim_1999_2021.zarr",
    "--geos_clim_path", "dataprocess/clim_pure/geos_weekly_ensmean_clim_1999_2021.zarr",
    "--obs_clim_path", "dataprocess/clim_pure/obs_weekly_clim_1999_2021.zarr",
    "--event_names", "pnw_heat_dome_2021", "sicily_heatwave_2021",
    "--output_dir", "ml_output_flowmulti/smoke_t2m_extremes_pure_2021_2022",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    smoke_main()


if __name__ == "__main__":
    main()
