#!/usr/bin/env python3
"""
Evaluate a catalog of historical extreme-event cases from saved forecast Zarrs.

The forecast products are weekly lead targets, so event "time series" are
weekly valid-time samples for lead weeks 3/4 around each event date. Metrics are
computed over event-specific land regional boxes using the saved ML/GEOS
ensembles. Calibrated BSS uses the logistic calibration table from the matrix
evaluation, and observed extreme thresholds come from the long-term observed
threshold NetCDF.
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from evaluate_matrix_suite_flow_finalv1_global import (
    add_map_overlays,
    configure_map_context,
    crps_map,
    inv_logit,
    logit,
    load_thresholds_from_file,
    make_map_subplots,
    season_name,
    select_grouped_map,
)


VARIABLES = {
    "pr": {
        "model": "model_pr",
        "geos": "geos_pr",
        "obs": "obs_pr",
        "units": "mm/day",
        "plot_units": "mm/day",
        "offset": 0.0,
    },
    "t2m": {
        "model": "model_t2m",
        "geos": "geos_t2m",
        "obs": "obs_t2m",
        "units": "K",
        "plot_units": "C",
        "offset": -273.15,
    },
}

DEFAULT_LEADS = [3, 4]
DEFAULT_TAIL_FRACTION = 0.10
DEFAULT_MAP_QUANTILE = 0.95
DEFAULT_NEIGHBORHOOD_RADIUS = 1


DEFAULT_EVENT_CATALOG = [
    # CONUS
    {
        "event_id": "conus_t2m_202106_pnw_heat_dome",
        "region": "conus",
        "region_label": "CONUS / USA",
        "variable": "t2m",
        "event_name": "Pacific Northwest heat dome",
        "event_start": "2021-06-25",
        "event_end": "2021-07-02",
        "bbox": (-125.0, -110.0, 40.0, 55.0),
        "source_url": "https://en.wikipedia.org/wiki/2021_Western_North_America_heat_wave",
        "source_note": "Record late-June/early-July 2021 western North America heat wave.",
    },
    {
        "event_id": "conus_t2m_202306_southern_us_heatwave",
        "region": "conus",
        "region_label": "CONUS / USA",
        "variable": "t2m",
        "event_name": "Southern U.S. heat wave",
        "event_start": "2023-06-20",
        "event_end": "2023-07-05",
        "bbox": (-106.0, -80.0, 25.0, 38.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_heat_waves",
        "source_note": "Broad late-June/early-July 2023 heat wave over the southern United States.",
    },
    {
        "event_id": "conus_pr_202108_tennessee_floods",
        "region": "conus",
        "region_label": "CONUS / USA",
        "variable": "pr",
        "event_name": "Middle Tennessee flash floods",
        "event_start": "2021-08-21",
        "event_end": "2021-08-22",
        "bbox": (-89.5, -86.0, 35.0, 37.0),
        "source_url": "https://en.wikipedia.org/wiki/2021_Tennessee_floods",
        "source_note": "Non-tropical August 2021 flash-flood rainfall over western Middle Tennessee.",
    },
    {
        "event_id": "conus_pr_202207_appalachian_kentucky_floods",
        "region": "conus",
        "region_label": "CONUS / USA",
        "variable": "pr",
        "event_name": "Appalachian / eastern Kentucky floods",
        "event_start": "2022-07-26",
        "event_end": "2022-08-01",
        "bbox": (-85.0, -80.0, 36.0, 39.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_Appalachian_floods",
        "source_note": "Stationary-frontal heavy-rain/flood event over eastern Kentucky and central Appalachia.",
    },
    {
        "event_id": "conus_pr_202301_california_atmospheric_rivers",
        "region": "conus",
        "region_label": "CONUS / USA",
        "variable": "pr",
        "event_name": "California atmospheric-river floods",
        "event_start": "2023-01-04",
        "event_end": "2023-01-11",
        "bbox": (-124.5, -117.0, 34.0, 41.5),
        "source_url": "https://en.wikipedia.org/wiki/2022%E2%80%932023_California_floods",
        "source_note": "Multiple atmospheric rivers produced broad California heavy rainfall and flooding.",
    },
    # Bangladesh
    {
        "event_id": "bangladesh_t2m_202304_heatwave",
        "region": "bangladesh",
        "region_label": "Bangladesh",
        "variable": "t2m",
        "event_name": "Bangladesh April heat wave",
        "event_start": "2023-04-14",
        "event_end": "2023-04-19",
        "bbox": (88.0, 93.0, 20.0, 27.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-asia-2023",
        "source_note": "Asia 2023 heat extremes; Bangladesh experienced severe April heat.",
    },
    {
        "event_id": "bangladesh_pr_202206_meghalaya_sylhet_downpours",
        "region": "bangladesh",
        "region_label": "Bangladesh",
        "variable": "pr",
        "event_name": "Meghalaya / Sylhet extreme downpours",
        "event_start": "2022-06-15",
        "event_end": "2022-06-18",
        "bbox": (88.5, 93.0, 23.5, 27.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_India%E2%80%93Bangladesh_floods",
        "source_note": "Focused Mawsynram/Cherrapunji and Sylhet-region monsoon-rainfall case.",
    },
    {
        "event_id": "bangladesh_pr_202206_sylhet_floods",
        "region": "bangladesh",
        "region_label": "Bangladesh",
        "variable": "pr",
        "event_name": "Sylhet / northeast Bangladesh floods",
        "event_start": "2022-06-15",
        "event_end": "2022-06-19",
        "bbox": (88.0, 93.0, 20.0, 27.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_India%E2%80%93Bangladesh_floods",
        "source_note": "June 2022 extreme rainfall/flooding in northeast India and Bangladesh.",
    },
    # India
    {
        "event_id": "india_t2m_202305_eastern_india_heatwave",
        "region": "india",
        "region_label": "India",
        "variable": "t2m",
        "event_name": "Eastern India late-spring heat wave",
        "event_start": "2023-04-15",
        "event_end": "2023-04-22",
        "bbox": (82.0, 90.0, 18.0, 27.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-asia-2023",
        "source_note": "Representative spring 2023 heat-stress episode over eastern India/Bangladesh region.",
    },
    {
        "event_id": "india_t2m_202204_india_pakistan_heatwave",
        "region": "india",
        "region_label": "India",
        "variable": "t2m",
        "event_name": "India-Pakistan spring heat wave",
        "event_start": "2022-04-25",
        "event_end": "2022-05-02",
        "bbox": (68.0, 88.0, 20.0, 32.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_India%E2%80%93Pakistan_heat_wave",
        "source_note": "Record early-season heat affecting northwest/central India and Pakistan.",
    },
    {
        "event_id": "india_pr_202107_maharashtra_floods",
        "region": "india",
        "region_label": "India",
        "variable": "pr",
        "event_name": "Maharashtra monsoon floods",
        "event_start": "2021-07-22",
        "event_end": "2021-07-27",
        "bbox": (72.0, 78.5, 15.0, 21.5),
        "source_url": "https://en.wikipedia.org/wiki/2021_Maharashtra_floods",
        "source_note": "Western India monsoon flooding and landslides during late July 2021.",
    },
    {
        "event_id": "india_pr_202307_north_india_floods",
        "region": "india",
        "region_label": "India",
        "variable": "pr",
        "event_name": "North India monsoon floods",
        "event_start": "2023-07-08",
        "event_end": "2023-07-11",
        "bbox": (74.0, 81.0, 28.0, 34.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_North_India_floods",
        "source_note": "July 2023 extreme monsoon rainfall over Himachal Pradesh/Uttarakhand/north India.",
    },
    # Pakistan
    {
        "event_id": "pakistan_t2m_202205_jacobabad_heatwave",
        "region": "pakistan",
        "region_label": "Pakistan",
        "variable": "t2m",
        "event_name": "Jacobabad / Sindh heat wave",
        "event_start": "2022-05-10",
        "event_end": "2022-05-17",
        "bbox": (66.0, 71.0, 25.0, 30.5),
        "source_url": "https://en.wikipedia.org/wiki/2022_India%E2%80%93Pakistan_heat_wave",
        "source_note": "Focused Pakistan hot-spot case during the spring 2022 South Asia heat wave.",
    },
    {
        "event_id": "pakistan_t2m_202204_india_pakistan_heatwave",
        "region": "pakistan",
        "region_label": "Pakistan",
        "variable": "t2m",
        "event_name": "Pakistan spring heat wave",
        "event_start": "2022-04-25",
        "event_end": "2022-05-02",
        "bbox": (62.0, 72.0, 24.0, 34.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_India%E2%80%93Pakistan_heat_wave",
        "source_note": "Nawabshah/Jacobabad region extreme heat during spring 2022.",
    },
    {
        "event_id": "pakistan_pr_202208_monsoon_floods",
        "region": "pakistan",
        "region_label": "Pakistan",
        "variable": "pr",
        "event_name": "Pakistan monsoon floods",
        "event_start": "2022-08-24",
        "event_end": "2022-08-28",
        "bbox": (60.0, 72.0, 23.0, 32.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_Pakistan_floods",
        "source_note": "Catastrophic 2022 Pakistan monsoon flooding.",
    },
    # Europe
    {
        "event_id": "europe_t2m_202108_mediterranean_heatwave",
        "region": "europe",
        "region_label": "Europe",
        "variable": "t2m",
        "event_name": "Mediterranean / Sicily heat wave",
        "event_start": "2021-08-09",
        "event_end": "2021-08-13",
        "bbox": (10.0, 20.0, 35.0, 42.0),
        "source_url": "https://en.wikipedia.org/wiki/2021_Mediterranean_heat_wave",
        "source_note": "August 2021 Mediterranean heat wave, including the Sicily European record-temperature episode.",
    },
    {
        "event_id": "europe_t2m_202207_uk_heatwave",
        "region": "europe",
        "region_label": "Europe",
        "variable": "t2m",
        "event_name": "United Kingdom record heat wave",
        "event_start": "2022-07-18",
        "event_end": "2022-07-20",
        "bbox": (-6.0, 2.0, 50.0, 56.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_United_Kingdom_heatwaves",
        "source_note": "July 2022 UK heat wave when temperatures exceeded 40C for the first time.",
    },
    {
        "event_id": "europe_t2m_202307_cerberus_heatwave",
        "region": "europe",
        "region_label": "Europe",
        "variable": "t2m",
        "event_name": "Southern Europe Cerberus heat wave",
        "event_start": "2023-07-17",
        "event_end": "2023-07-24",
        "bbox": (-10.0, 25.0, 36.0, 47.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_European_heatwaves",
        "source_note": "July 2023 severe southern European heat wave.",
    },
    {
        "event_id": "europe_pr_202305_emilia_romagna_floods",
        "region": "europe",
        "region_label": "Europe",
        "variable": "pr",
        "event_name": "Emilia-Romagna floods",
        "event_start": "2023-05-16",
        "event_end": "2023-05-18",
        "bbox": (10.0, 13.5, 43.5, 45.5),
        "source_url": "https://en.wikipedia.org/wiki/2023_Emilia-Romagna_floods",
        "source_note": "Northern Italy heavy-rain/flood event during May 2023.",
    },
    {
        "event_id": "europe_pr_202308_slovenia_floods",
        "region": "europe",
        "region_label": "Europe",
        "variable": "pr",
        "event_name": "Slovenia floods",
        "event_start": "2023-08-03",
        "event_end": "2023-08-05",
        "bbox": (13.0, 17.0, 45.5, 47.5),
        "source_url": "https://en.wikipedia.org/wiki/2023_Slovenia_floods",
        "source_note": "Early-August 2023 heavy-rain/flood event over Slovenia and nearby Alpine regions.",
    },
    {
        "event_id": "europe_pr_202107_western_europe_floods",
        "region": "europe",
        "region_label": "Europe",
        "variable": "pr",
        "event_name": "Western Europe floods",
        "event_start": "2021-07-12",
        "event_end": "2021-07-16",
        "bbox": (2.0, 10.0, 48.0, 52.0),
        "source_url": "https://en.wikipedia.org/wiki/2021_European_floods",
        "source_note": "July 2021 extreme rainfall/flooding over western Germany/Belgium/Netherlands.",
    },
    # Australia
    {
        "event_id": "australia_t2m_202201_onslow_heat",
        "region": "australia",
        "region_label": "Australia",
        "variable": "t2m",
        "event_name": "Western Australia Onslow extreme heat",
        "event_start": "2022-01-12",
        "event_end": "2022-01-15",
        "bbox": (112.0, 124.0, -28.0, -17.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_heat_waves",
        "source_note": "Onslow, Western Australia reached 50.7C in January 2022.",
    },
    {
        "event_id": "australia_pr_202202_eastern_australia_floods",
        "region": "australia",
        "region_label": "Australia",
        "variable": "pr",
        "event_name": "Eastern Australia floods",
        "event_start": "2022-02-26",
        "event_end": "2022-03-04",
        "bbox": (145.0, 154.0, -38.0, -25.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_eastern_Australia_floods",
        "source_note": "Late February/March 2022 Queensland/New South Wales flood disaster.",
    },
    # Africa and subregions
    {
        "event_id": "africa_t2m_202307_north_africa_heat",
        "region": "africa",
        "region_label": "Africa",
        "variable": "t2m",
        "event_name": "North Africa / Mediterranean heat wave",
        "event_start": "2023-07-17",
        "event_end": "2023-07-25",
        "bbox": (-10.0, 40.0, 20.0, 38.0),
        "source_url": "https://www.axios.com/2023/07/18/heat-wave-temperatures-us-europe-asia",
        "source_note": "July 2023 intense heat affected Europe, the Middle East and North Africa.",
    },
    {
        "event_id": "africa_pr_202309_libya_storm_daniel",
        "region": "africa",
        "region_label": "Africa",
        "variable": "pr",
        "event_name": "Libya Storm Daniel floods",
        "event_start": "2023-09-10",
        "event_end": "2023-09-12",
        "bbox": (19.0, 25.0, 30.0, 34.0),
        "source_url": "https://en.wikipedia.org/wiki/Storm_Daniel",
        "source_note": "Storm Daniel caused catastrophic flooding in eastern Libya.",
    },
    {
        "event_id": "south_america_t2m_202308_winter_heatwave",
        "region": "south_america",
        "region_label": "South America",
        "variable": "t2m",
        "event_name": "South America winter heat wave",
        "event_start": "2023-08-01",
        "event_end": "2023-08-12",
        "bbox": (-70.0, -45.0, -35.0, -10.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_heat_waves",
        "source_note": "Unseasonal South America winter heat wave in August 2023.",
    },
    {
        "event_id": "south_america_pr_202202_petropolis_floods",
        "region": "south_america",
        "region_label": "South America",
        "variable": "pr",
        "event_name": "Petrópolis / Rio de Janeiro floods",
        "event_start": "2022-02-15",
        "event_end": "2022-02-16",
        "bbox": (-47.0, -40.0, -24.0, -20.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_Petr%C3%B3polis_floods",
        "source_note": "February 2022 extreme rainfall and flooding in Petrópolis, Brazil.",
    },
    {
        "event_id": "amazon_basin_t2m_202309_amazon_heat_drought",
        "region": "amazon_basin",
        "region_label": "Amazon Basin",
        "variable": "t2m",
        "event_name": "Amazon drought / heat",
        "event_start": "2023-09-20",
        "event_end": "2023-10-05",
        "bbox": (-75.0, -50.0, -15.0, 5.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-latin-america-and-caribbean-2023",
        "source_note": "2023 Amazon drought/heat conditions during exceptional regional warmth.",
    },
    {
        "event_id": "amazon_basin_pr_202303_western_amazon_floods",
        "region": "amazon_basin",
        "region_label": "Amazon Basin",
        "variable": "pr",
        "event_name": "Western Amazon / Acre floods",
        "event_start": "2023-03-20",
        "event_end": "2023-03-28",
        "bbox": (-75.0, -55.0, -12.0, 5.0),
        "source_url": "https://floodlist.com/america/brazil-acre-floods-march-2023",
        "source_note": "March 2023 floods in western Amazon/Acre region.",
    },
    {
        "event_id": "sahel_west_africa_t2m_202304_sahel_heat",
        "region": "sahel_west_africa",
        "region_label": "Sahel / West Africa",
        "variable": "t2m",
        "event_name": "Sahel spring heat",
        "event_start": "2023-04-01",
        "event_end": "2023-04-15",
        "bbox": (-15.0, 25.0, 10.0, 20.0),
        "source_url": "https://www.worldweatherattribution.org/",
        "source_note": "Representative Sahel hot-season extreme; catalog entry can be refined.",
    },
    {
        "event_id": "sahel_west_africa_pr_202208_west_africa_floods",
        "region": "sahel_west_africa",
        "region_label": "Sahel / West Africa",
        "variable": "pr",
        "event_name": "West Africa / Nigeria floods",
        "event_start": "2022-08-15",
        "event_end": "2022-09-15",
        "bbox": (-5.0, 15.0, 8.0, 16.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_Nigeria_floods",
        "source_note": "Major 2022 West Africa/Nigeria flooding season.",
    },
    {
        "event_id": "east_africa_horn_t2m_202203_horn_heat_drought",
        "region": "east_africa_horn",
        "region_label": "East Africa / Horn",
        "variable": "t2m",
        "event_name": "Horn of Africa heat/drought episode",
        "event_start": "2022-03-01",
        "event_end": "2022-03-15",
        "bbox": (35.0, 50.0, -5.0, 12.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-africa-2022",
        "source_note": "Horn of Africa drought/heat stress during 2022 failed rainy seasons.",
    },
    {
        "event_id": "east_africa_horn_pr_202311_horn_floods",
        "region": "east_africa_horn",
        "region_label": "East Africa / Horn",
        "variable": "pr",
        "event_name": "Horn of Africa floods",
        "event_start": "2023-11-01",
        "event_end": "2023-11-15",
        "bbox": (35.0, 50.0, -5.0, 12.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-africa-2023",
        "source_note": "Late-2023 Horn of Africa floods after prolonged drought.",
    },
    {
        "event_id": "southern_africa_t2m_202301_southern_africa_heat",
        "region": "southern_africa",
        "region_label": "Southern Africa",
        "variable": "t2m",
        "event_name": "Southern Africa summer heat",
        "event_start": "2023-01-20",
        "event_end": "2023-01-31",
        "bbox": (18.0, 35.0, -30.0, -15.0),
        "source_url": "https://wmo.int/publication-series/state-of-climate-africa-2023",
        "source_note": "Representative austral-summer southern Africa heat episode.",
    },
    {
        "event_id": "southern_africa_pr_202204_kwazulu_natal_floods",
        "region": "southern_africa",
        "region_label": "Southern Africa",
        "variable": "pr",
        "event_name": "KwaZulu-Natal floods",
        "event_start": "2022-04-11",
        "event_end": "2022-04-13",
        "bbox": (29.0, 32.5, -31.0, -28.0),
        "source_url": "https://en.wikipedia.org/wiki/2022_KwaZulu-Natal_floods",
        "source_note": "Non-tropical April 2022 heavy-rain/flood disaster around Durban/KwaZulu-Natal.",
    },
    # Mediterranean/Middle East
    {
        "event_id": "mediterranean_middle_east_t2m_202307_heatwave",
        "region": "mediterranean_middle_east",
        "region_label": "Mediterranean / Middle East",
        "variable": "t2m",
        "event_name": "Mediterranean / Middle East heat wave",
        "event_start": "2023-07-17",
        "event_end": "2023-07-25",
        "bbox": (25.0, 55.0, 25.0, 42.0),
        "source_url": "https://www.axios.com/2023/07/18/heat-wave-temperatures-us-europe-asia",
        "source_note": "July 2023 intense heat over the Mediterranean/Middle East.",
    },
    {
        "event_id": "mediterranean_middle_east_pr_202309_storm_daniel",
        "region": "mediterranean_middle_east",
        "region_label": "Mediterranean / Middle East",
        "variable": "pr",
        "event_name": "Storm Daniel Mediterranean floods",
        "event_start": "2023-09-05",
        "event_end": "2023-09-12",
        "bbox": (15.0, 30.0, 30.0, 42.0),
        "source_url": "https://en.wikipedia.org/wiki/Storm_Daniel",
        "source_note": "Storm Daniel affected Greece/Libya/eastern Mediterranean with extreme rainfall.",
    },
    # Southeast/East Asia and Caribbean
    {
        "event_id": "southeast_asia_t2m_202304_heatwave",
        "region": "southeast_asia",
        "region_label": "Southeast Asia",
        "variable": "t2m",
        "event_name": "Southeast Asia April heat wave",
        "event_start": "2023-04-15",
        "event_end": "2023-04-30",
        "bbox": (95.0, 110.0, 5.0, 25.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_heat_waves",
        "source_note": "April 2023 record heat across parts of South and Southeast Asia.",
    },
    {
        "event_id": "east_asia_t2m_202306_china_heatwave",
        "region": "east_asia",
        "region_label": "East Asia",
        "variable": "t2m",
        "event_name": "China / North China heat wave",
        "event_start": "2023-06-22",
        "event_end": "2023-06-25",
        "bbox": (105.0, 125.0, 30.0, 45.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_China_heat_wave",
        "source_note": "June 2023 record heat in Beijing/northern China.",
    },
    {
        "event_id": "east_asia_pr_202107_henan_floods",
        "region": "east_asia",
        "region_label": "East Asia",
        "variable": "pr",
        "event_name": "Henan / Zhengzhou extreme rainfall",
        "event_start": "2021-07-17",
        "event_end": "2021-07-22",
        "bbox": (112.0, 115.0, 33.0, 36.0),
        "source_url": "https://en.wikipedia.org/wiki/2021_Henan_floods",
        "source_note": "July 2021 extreme rainfall/flooding around Zhengzhou/Henan; retained as a non-landfall East Asia PR case.",
    },
    {
        "event_id": "central_america_caribbean_t2m_202306_caribbean_heat",
        "region": "central_america_caribbean",
        "region_label": "Central America / Caribbean",
        "variable": "t2m",
        "event_name": "Caribbean heat wave",
        "event_start": "2023-06-06",
        "event_end": "2023-06-10",
        "bbox": (-85.0, -60.0, 10.0, 25.0),
        "source_url": "https://en.wikipedia.org/wiki/2023_Caribbean_heat_wave",
        "source_note": "Early June 2023 Caribbean heat wave and high heat index.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate historical event cases from saved global forecast Zarrs.")
    parser.add_argument("--forecast_dir", type=str, default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50")
    parser.add_argument(
        "--threshold_file",
        type=str,
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "matrix_eval_global_2021_2023_land_obsclim_chunked/event_thresholds_and_frequencies.nc"
        ),
    )
    parser.add_argument(
        "--calibration_params",
        type=str,
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "matrix_eval_global_2021_2023_land_obsclim_chunked/bss_calibration_params.csv"
        ),
    )
    parser.add_argument("--land_mask_file", type=str, default="ml_model/land_ocean_mask_v6.pt")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023",
    )
    parser.add_argument("--event_catalog", type=str, default="default", help="default or path to CSV/JSON event catalog.")
    parser.add_argument("--regions", type=str, default="all")
    parser.add_argument("--variables", type=str, default="pr,t2m")
    parser.add_argument("--leads", type=str, default="3,4")
    parser.add_argument("--progression_leads", type=str, default="1,2,3,4")
    parser.add_argument(
        "--regional_weighting",
        choices=("uniform", "area"),
        default="uniform",
        help="Regional/event aggregation weights. Uniform is usually easier to interpret for local event diagnostics.",
    )
    parser.add_argument(
        "--tail_fraction",
        type=float,
        default=DEFAULT_TAIL_FRACTION,
        help="Fraction of event-box land grid cells used for the top-tail intensity time series.",
    )
    parser.add_argument("--extreme_quantile_pr", type=float, default=0.95)
    parser.add_argument("--extreme_quantile_t2m", type=float, default=0.95)
    parser.add_argument(
        "--map_quantile",
        type=float,
        default=DEFAULT_MAP_QUANTILE,
        help="Ensemble upper quantile shown in event maps/time series; default 0.95.",
    )
    parser.add_argument(
        "--neighborhood_radius",
        type=int,
        default=DEFAULT_NEIGHBORHOOD_RADIUS,
        help="Grid-cell radius for neighborhood exceedance probability. 0 disables neighborhood expansion.",
    )
    parser.add_argument("--pr_min_threshold", type=float, default=5.0)
    parser.add_argument("--timeseries_window_days", type=int, default=42)
    parser.add_argument("--event_tolerance_days", type=int, default=10)
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--map_features", choices=("auto", "cartopy", "plain"), default="auto")
    parser.add_argument("--county_boundaries", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_list(text, cast=str):
    return [cast(item.strip()) for item in str(text or "").split(",") if item.strip()]


def parse_bbox(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
    else:
        text = str(value).strip()
        if text.startswith("["):
            vals = json.loads(text)
        else:
            vals = [item.strip() for item in text.strip("()[]").split(",") if item.strip()]
    if len(vals) != 4:
        raise ValueError(f"Bounding box must have four values lon_min,lon_max,lat_min,lat_max; got {value!r}")
    return tuple(float(item) for item in vals)


def lon_to_180(lons):
    return ((np.asarray(lons, dtype=np.float64) + 180.0) % 360.0) - 180.0


def bbox_mask(lons, lats, bbox):
    lon_min, lon_max, lat_min, lat_max = [float(x) for x in bbox]
    lon2d, lat2d = np.meshgrid(lon_to_180(lons), np.asarray(lats, dtype=np.float64))
    if lon_min <= lon_max:
        lon_ok = (lon2d >= lon_min) & (lon2d <= lon_max)
    else:
        lon_ok = (lon2d >= lon_min) | (lon2d <= lon_max)
    return lon_ok & (lat2d >= lat_min) & (lat2d <= lat_max)


def load_land_mask(path, shape):
    if not path or not os.path.exists(path):
        print(f"⚠️ Land mask missing ({path}); using all grid points.")
        return np.ones(shape, dtype=bool), None
    import torch

    cached = torch.load(path, map_location="cpu", weights_only=True)
    if "is_land" in cached:
        land = np.asarray(cached["is_land"], dtype=bool).squeeze()
    elif "land_mask" in cached:
        land = np.asarray(cached["land_mask"], dtype=bool).squeeze()
    else:
        raise ValueError(f"{path} is missing is_land or land_mask")
    if land.shape != shape:
        raise ValueError(f"Land mask shape {land.shape} does not match grid shape {shape}")
    return land, os.path.abspath(path)


def load_event_catalog(path_or_default):
    if str(path_or_default).lower() == "default":
        return pd.DataFrame(DEFAULT_EVENT_CATALOG)
    if path_or_default.endswith(".json"):
        with open(path_or_default) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path_or_default)


def normalize_catalog(catalog, regions, variables, start_year, end_year):
    catalog = catalog.copy()
    catalog["event_start"] = pd.to_datetime(catalog["event_start"]).dt.normalize()
    catalog["event_end"] = pd.to_datetime(catalog["event_end"]).dt.normalize()
    catalog["event_center"] = catalog["event_start"] + (catalog["event_end"] - catalog["event_start"]) / 2
    if regions != ["all"]:
        catalog = catalog[catalog["region"].isin(regions)]
    catalog = catalog[catalog["variable"].isin(variables)]
    catalog = catalog[(catalog["event_center"].dt.year >= start_year) & (catalog["event_center"].dt.year <= end_year)]
    if catalog.empty:
        raise ValueError("No events selected after filtering.")
    return catalog.reset_index(drop=True)


def area_weights(lats):
    return np.clip(np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64))), 0.0, None)[:, None]


def region_weights(lats, mask, mode="uniform"):
    mask = np.asarray(mask, dtype=bool)
    if mode == "area":
        return area_weights(lats) * mask.astype(np.float64)
    if mode == "uniform":
        return mask.astype(np.float64)
    raise ValueError(f"Unknown regional weighting mode: {mode}")


def weighted_mean(field, weights):
    field = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    return float(np.sum(field[finite] * weights[finite]) / np.sum(weights[finite]))


def weighted_top_mean(field, weights, fraction=DEFAULT_TAIL_FRACTION):
    field = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    values = field[finite]
    w = weights[finite]
    order = np.argsort(values)[::-1]
    values = values[order]
    w = w[order]
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        return np.nan
    target_weight = max(float(fraction), 1e-6) * total_weight
    cumulative = np.cumsum(w)
    keep = cumulative <= target_weight
    if not keep.any():
        keep[0] = True
    elif keep.sum() < len(keep) and cumulative[keep].max(initial=0.0) < target_weight:
        keep[keep.sum()] = True
    return float(np.sum(values[keep] * w[keep]) / np.sum(w[keep]))


def regional_member_values(ensemble, weights):
    values = []
    for member in np.asarray(ensemble):
        values.append(weighted_mean(member, weights))
    return np.asarray(values, dtype=np.float64)


def regional_member_tail_values(ensemble, weights, fraction=DEFAULT_TAIL_FRACTION):
    values = []
    for member in np.asarray(ensemble):
        values.append(weighted_top_mean(member, weights, fraction=fraction))
    return np.asarray(values, dtype=np.float64)


def open_forecast_grid(forecast_dir, start_year, end_year):
    for year in range(start_year, end_year + 1):
        path = os.path.join(forecast_dir, f"{year}.zarr")
        if os.path.exists(path):
            ds = xr.open_zarr(path, consolidated=False, chunks=None)
            try:
                return ds["lat"].values, ds["lon"].values
            finally:
                ds.close()
    raise FileNotFoundError(f"No YEAR.zarr stores found under {forecast_dir}")


def valid_times_for_dataset(ds, init_idx, init_time, lead_values):
    if "valid_time" in ds:
        return pd.to_datetime(ds["valid_time"].isel(init=init_idx).values).normalize()
    return pd.to_datetime(
        [init_time + pd.to_timedelta(int(lead) * 7, unit="D") for lead in lead_values]
    ).normalize()


def target_window_for_valid_time(valid_time):
    """Treat saved weekly targets as 7-day windows ending on valid_time."""
    valid_time = pd.Timestamp(valid_time).normalize()
    return valid_time - pd.Timedelta(days=6), valid_time


def build_sample_record(zarr_path, zarr_year, init_idx, lead_idx, lead_value, init_time, valid_time, event):
    event_start_core = pd.Timestamp(event["event_start"]).normalize()
    event_end_core = pd.Timestamp(event["event_end"]).normalize()
    target_start, target_end = target_window_for_valid_time(valid_time)
    event_overlap_days = inclusive_overlap_days(target_start, target_end, event_start_core, event_end_core)
    return {
        "zarr_path": zarr_path,
        "zarr_year": int(zarr_year),
        "init_idx": int(init_idx),
        "lead_idx": int(lead_idx),
        "lead": int(lead_value),
        "init_time": pd.Timestamp(init_time).normalize(),
        "valid_time": pd.Timestamp(valid_time).normalize(),
        "target_window_start": target_start,
        "target_window_end": target_end,
        "event_overlap_days": int(event_overlap_days),
        "event_overlap_fraction": float(event_overlap_days / 7.0),
        "event_distance_days": interval_distance_days(target_start, target_end, event_start_core, event_end_core),
        "in_event_window": bool(event_overlap_days > 0),
    }


def inclusive_overlap_days(start_a, end_a, start_b, end_b):
    start = max(pd.Timestamp(start_a).normalize(), pd.Timestamp(start_b).normalize())
    end = min(pd.Timestamp(end_a).normalize(), pd.Timestamp(end_b).normalize())
    if end < start:
        return 0
    return int((end - start).days) + 1


def interval_distance_days(start_a, end_a, start_b, end_b):
    start_a = pd.Timestamp(start_a).normalize()
    end_a = pd.Timestamp(end_a).normalize()
    start_b = pd.Timestamp(start_b).normalize()
    end_b = pd.Timestamp(end_b).normalize()
    if end_a < start_b:
        return int((start_b - end_a).days)
    if end_b < start_a:
        return int((start_a - end_b).days)
    return 0


def find_candidate_samples(forecast_dir, event, leads, timeseries_window_days, event_tolerance_days):
    center = pd.Timestamp(event["event_center"]).normalize()
    ts_start = center - pd.Timedelta(days=int(timeseries_window_days))
    ts_end = center + pd.Timedelta(days=int(timeseries_window_days))
    event_start_core = pd.Timestamp(event["event_start"]).normalize()
    event_end_core = pd.Timestamp(event["event_end"]).normalize()
    event_start_tol = event_start_core - pd.Timedelta(days=int(event_tolerance_days))
    event_end_tol = event_end_core + pd.Timedelta(days=int(event_tolerance_days))
    year_candidates = sorted(set([center.year - 1, center.year, center.year + 1]))
    samples = []
    for year in year_candidates:
        path = os.path.join(forecast_dir, f"{year}.zarr")
        if not os.path.exists(path):
            continue
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        try:
            init_values = pd.to_datetime(ds["init"].values).normalize()
            lead_values = ds["lead"].values
            for init_idx, init_time in enumerate(init_values):
                valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
                for lead_idx, lead_value in enumerate(lead_values):
                    lead_value = int(lead_value)
                    if lead_value not in leads:
                        continue
                    valid_time = pd.Timestamp(valid_values[lead_idx]).normalize()
                    target_start, target_end = target_window_for_valid_time(valid_time)
                    if target_end >= ts_start and target_start <= ts_end:
                        sample = build_sample_record(
                            path,
                            year,
                            init_idx,
                            lead_idx,
                            lead_value,
                            init_time,
                            valid_time,
                            event,
                        )
                        tolerance_overlap_days = inclusive_overlap_days(target_start, target_end, event_start_tol, event_end_tol)
                        sample["tolerance_overlap_days"] = int(tolerance_overlap_days)
                        sample["in_event_tolerance_window"] = bool(tolerance_overlap_days > 0)
                        samples.append(sample)
        finally:
            ds.close()
    return samples


def fixed_init_progression_samples(selected_samples, event, progression_leads):
    samples = []
    seen = set()
    for selected in selected_samples:
        key = (selected["zarr_path"], int(selected["zarr_year"]), int(selected["init_idx"]))
        if key in seen:
            continue
        seen.add(key)
        zarr_path, zarr_year, init_idx = key
        ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
        try:
            init_values = pd.to_datetime(ds["init"].values).normalize()
            init_time = pd.Timestamp(init_values[init_idx]).normalize()
            lead_values = ds["lead"].values
            valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
            for lead_idx, lead_value in enumerate(lead_values):
                lead_value = int(lead_value)
                if lead_value not in progression_leads:
                    continue
                sample = build_sample_record(
                    zarr_path,
                    zarr_year,
                    init_idx,
                    lead_idx,
                    lead_value,
                    init_time,
                    pd.Timestamp(valid_values[lead_idx]).normalize(),
                    event,
                )
                sample["selection_mode"] = "fixed_init_progression"
                samples.append(sample)
        finally:
            ds.close()
    return sorted(samples, key=lambda item: (item["init_time"], item["lead"]))


def choose_event_samples(samples, leads):
    selected = []
    by_lead = defaultdict(list)
    for sample in samples:
        by_lead[int(sample["lead"])].append(sample)
    for lead in leads:
        lead_samples = by_lead.get(int(lead), [])
        event_window = [sample for sample in lead_samples if sample["in_event_window"]]
        tolerance_window = [sample for sample in lead_samples if sample.get("in_event_tolerance_window")]
        if event_window:
            pool = event_window
            selection_mode = "event_window_overlap"
        elif tolerance_window:
            pool = tolerance_window
            selection_mode = "event_tolerance_overlap"
        else:
            pool = lead_samples
            selection_mode = "closest"
        if not pool:
            continue
        pool = sorted(
            pool,
            key=lambda item: (-int(item.get("event_overlap_days", 0)), item["event_distance_days"], item["valid_time"]),
        )
        closest = dict(pool[0])
        closest["selected_for_event_metrics"] = True
        closest["selection_mode"] = selection_mode
        selected.append(closest)
    return selected


def load_calibration_params(path):
    if not path or not os.path.exists(path):
        print(f"⚠️ Calibration params missing ({path}); calibrated BSS will use raw probabilities.")
        return {}
    table = pd.read_csv(path)
    out = {}
    for _, row in table.iterrows():
        key = (str(row["variable"]), str(row["source"]), int(row["holdout_year"]), int(row["lead"]), str(row["season"]))
        out[key] = {
            "intercept": float(row["intercept"]),
            "slope": float(row["slope"]),
            "method": str(row.get("method", "logistic")),
        }
    return out


def apply_calibration(prob, calibrator, eps=1e-4):
    prob = np.asarray(prob, dtype=np.float64)
    if not calibrator:
        return np.clip(prob, 0.0, 1.0)
    return np.clip(
        inv_logit(float(calibrator.get("intercept", 0.0)) + float(calibrator.get("slope", 1.0)) * logit(prob, eps)),
        0.0,
        1.0,
    )


def skill_pct(model_value, geos_value):
    if not np.isfinite(model_value) or not np.isfinite(geos_value) or abs(geos_value) <= 1e-12:
        return np.nan
    return float(100.0 * (1.0 - model_value / geos_value))


def shift_no_wrap_bool(arr, dy, dx):
    """Shift a boolean (..., lat, lon) array without longitude wrapping."""
    arr = np.asarray(arr, dtype=bool)
    out = np.zeros_like(arr, dtype=bool)
    nlat, nlon = arr.shape[-2:]
    src_lat0 = max(0, -dy)
    src_lat1 = nlat - max(0, dy)
    dst_lat0 = max(0, dy)
    dst_lat1 = nlat - max(0, -dy)
    src_lon0 = max(0, -dx)
    src_lon1 = nlon - max(0, dx)
    dst_lon0 = max(0, dx)
    dst_lon1 = nlon - max(0, -dx)
    if src_lat0 >= src_lat1 or src_lon0 >= src_lon1:
        return out
    out[..., dst_lat0:dst_lat1, dst_lon0:dst_lon1] = arr[..., src_lat0:src_lat1, src_lon0:src_lon1]
    return out


def neighborhood_event_probability(ensemble, threshold, radius=DEFAULT_NEIGHBORHOOD_RADIUS):
    """Probability that an ensemble member exceeds threshold within a local grid-cell neighborhood."""
    radius = int(radius)
    threshold = np.asarray(threshold, dtype=np.float32)
    ensemble = np.asarray(ensemble, dtype=np.float32)
    exceed = np.isfinite(ensemble) & np.isfinite(threshold)[None, :, :] & (ensemble >= threshold[None, :, :])
    if radius <= 0:
        return np.nanmean(exceed.astype(np.float32), axis=0).astype(np.float32)
    neighborhood = np.zeros_like(exceed, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighborhood |= shift_no_wrap_bool(exceed, dy, dx)
    return np.nanmean(neighborhood.astype(np.float32), axis=0).astype(np.float32)


def robust_limits(fields, center_zero=False, lower=5, upper=95, fallback=(-1.0, 1.0)):
    finite_parts = []
    for field in fields:
        arr = np.asarray(field, dtype=np.float64)
        vals = arr[np.isfinite(arr)]
        if vals.size:
            finite_parts.append(vals)
    if not finite_parts:
        return fallback
    vals = np.concatenate(finite_parts)
    if center_zero:
        vmax = max(float(np.nanpercentile(np.abs(vals), upper)), 1e-6)
        return -vmax, vmax
    vmin, vmax = np.nanpercentile(vals, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) <= 1e-12:
        mid = float(np.nanmean(vals)) if vals.size else 0.0
        return mid - 1.0, mid + 1.0
    return float(vmin), float(vmax)


def evaluate_ensemble_metrics(
    ensemble,
    obs,
    threshold,
    obs_event_freq,
    weights,
    calibrator=None,
    tail_fraction=DEFAULT_TAIL_FRACTION,
    map_quantile=DEFAULT_MAP_QUANTILE,
    neighborhood_radius=DEFAULT_NEIGHBORHOOD_RADIUS,
):
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    ens_mean = np.nanmean(ensemble, axis=0)
    ens_upper = np.nanquantile(ensemble, float(map_quantile), axis=0).astype(np.float32)
    err = ens_mean - obs
    valid_weights = np.where(np.isfinite(obs) & np.isfinite(ens_mean) & np.isfinite(threshold), weights, 0.0)
    prob = np.nanmean(ensemble >= threshold[None, :, :], axis=0).astype(np.float64, copy=False)
    prob_cal = apply_calibration(prob, calibrator)
    prob_neighborhood = neighborhood_event_probability(
        ensemble,
        threshold,
        radius=neighborhood_radius,
    ).astype(np.float64, copy=False)
    event = obs >= threshold
    brier = (prob - event) ** 2
    brier_cal = (prob_cal - event) ** 2
    ref_brier = (obs_event_freq - event) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        bss_map = np.where(ref_brier > 1e-12, 1.0 - brier / ref_brier, np.nan)
        bss_cal_map = np.where(ref_brier > 1e-12, 1.0 - brier_cal / ref_brier, np.nan)
    crps = crps_map(ensemble, obs)
    sse = weighted_mean(err * err, valid_weights)
    bs = weighted_mean(brier, valid_weights)
    bs_cal = weighted_mean(brier_cal, valid_weights)
    ref_bs = weighted_mean(ref_brier, valid_weights)
    regional_members = regional_member_values(ensemble, valid_weights)
    regional_tail_members = regional_member_tail_values(ensemble, valid_weights, fraction=tail_fraction)
    obs_extreme_weights = np.where(event, valid_weights, 0.0)
    return {
        "mean": weighted_mean(ens_mean, valid_weights),
        "spread_region_mean": float(np.nanstd(regional_members)) if regional_members.size else np.nan,
        "q10_region_mean": float(np.nanquantile(regional_members, 0.10)) if regional_members.size else np.nan,
        "q90_region_mean": float(np.nanquantile(regional_members, 0.90)) if regional_members.size else np.nan,
        "q95_region_mean": float(np.nanquantile(regional_members, 0.95)) if regional_members.size else np.nan,
        "upper_quantile_mean": weighted_mean(ens_upper, valid_weights),
        "upper_quantile_tail_mean": weighted_top_mean(ens_upper, valid_weights, fraction=tail_fraction),
        "tail_mean": weighted_top_mean(ens_mean, valid_weights, fraction=tail_fraction),
        "tail_spread": float(np.nanstd(regional_tail_members)) if regional_tail_members.size else np.nan,
        "tail_q10": float(np.nanquantile(regional_tail_members, 0.10)) if regional_tail_members.size else np.nan,
        "tail_q90": float(np.nanquantile(regional_tail_members, 0.90)) if regional_tail_members.size else np.nan,
        "tail_q95": float(np.nanquantile(regional_tail_members, 0.95)) if regional_tail_members.size else np.nan,
        "rmse": float(np.sqrt(sse)) if np.isfinite(sse) else np.nan,
        "mae": weighted_mean(np.abs(err), valid_weights),
        "bias": weighted_mean(err, valid_weights),
        "crps": weighted_mean(crps, valid_weights),
        "crps_on_obs_extreme": weighted_mean(crps, obs_extreme_weights),
        "spread_grid": weighted_mean(np.nanstd(ensemble.astype(np.float64, copy=False), axis=0), valid_weights),
        "event_probability": weighted_mean(prob, valid_weights),
        "event_probability_calibrated": weighted_mean(prob_cal, valid_weights),
        "event_probability_neighborhood": weighted_mean(prob_neighborhood, valid_weights),
        "event_probability_top_tail": weighted_top_mean(prob, valid_weights, fraction=tail_fraction),
        "event_probability_calibrated_top_tail": weighted_top_mean(prob_cal, valid_weights, fraction=tail_fraction),
        "event_probability_neighborhood_top_tail": weighted_top_mean(
            prob_neighborhood,
            valid_weights,
            fraction=tail_fraction,
        ),
        "event_probability_on_obs_extreme": weighted_mean(prob, np.where(event, valid_weights, 0.0)),
        "event_probability_on_obs_nonextreme": weighted_mean(prob, np.where(~event, valid_weights, 0.0)),
        "event_probability_neighborhood_on_obs_extreme": weighted_mean(prob_neighborhood, obs_extreme_weights),
        "event_probability_neighborhood_on_obs_nonextreme": weighted_mean(
            prob_neighborhood,
            np.where(~event, valid_weights, 0.0),
        ),
        "cal_event_probability_on_obs_extreme": weighted_mean(prob_cal, np.where(event, valid_weights, 0.0)),
        "cal_event_probability_on_obs_nonextreme": weighted_mean(prob_cal, np.where(~event, valid_weights, 0.0)),
        "event_area_fraction_prob50": weighted_mean((prob >= 0.5).astype(np.float32), valid_weights),
        "cal_event_area_fraction_prob50": weighted_mean((prob_cal >= 0.5).astype(np.float32), valid_weights),
        "neighborhood_event_area_fraction_prob50": weighted_mean(
            (prob_neighborhood >= 0.5).astype(np.float32),
            valid_weights,
        ),
        "brier": bs,
        "brier_calibrated": bs_cal,
        "ref_brier": ref_bs,
        "bss": 1.0 - bs / ref_bs if np.isfinite(ref_bs) and ref_bs > 1e-12 else np.nan,
        "calibrated_bss": 1.0 - bs_cal / ref_bs if np.isfinite(ref_bs) and ref_bs > 1e-12 else np.nan,
        "members_region_mean": regional_members,
        "mean_map": ens_mean,
        "upper_quantile_map": ens_upper,
        "crps_map": crps,
        "bss_map": bss_map,
        "bss_cal_map": bss_cal_map,
        "prob_map": prob,
        "prob_cal_map": prob_cal,
        "prob_neighborhood_map": prob_neighborhood,
    }


def evaluate_sample(sample, event, thresholds, obs_clim, calibrators, weights):
    variable = str(event["variable"])
    spec = VARIABLES[variable]
    ds = xr.open_zarr(sample["zarr_path"], consolidated=False, chunks=None)
    try:
        obs = ds[spec["obs"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
        model_ens = ds[spec["model"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
        geos_ens = ds[spec["geos"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
    finally:
        ds.close()
    valid_time = pd.Timestamp(sample["valid_time"])
    threshold = select_grouped_map(thresholds[variable], valid_time)
    obs_event_freq = select_grouped_map(obs_clim[variable], valid_time)
    season = season_name(valid_time.month)
    # Matrix calibration was trained/evaluated by forecast Zarr year, not by
    # lead-valid year. For cross-year lead cases, use the same holdout key the
    # matrix suite used, otherwise the lookup silently falls back to raw BSS.
    calibration_year = int(sample["zarr_year"])
    model_cal = calibrators.get((variable, "model", calibration_year, int(sample["lead"]), season))
    geos_cal = calibrators.get((variable, "geos", calibration_year, int(sample["lead"]), season))
    model = evaluate_ensemble_metrics(
        model_ens,
        obs,
        threshold,
        obs_event_freq,
        weights,
        calibrator=model_cal,
        tail_fraction=DEFAULT_TAIL_FRACTION,
        map_quantile=DEFAULT_MAP_QUANTILE,
        neighborhood_radius=DEFAULT_NEIGHBORHOOD_RADIUS,
    )
    geos = evaluate_ensemble_metrics(
        geos_ens,
        obs,
        threshold,
        obs_event_freq,
        weights,
        calibrator=geos_cal,
        tail_fraction=DEFAULT_TAIL_FRACTION,
        map_quantile=DEFAULT_MAP_QUANTILE,
        neighborhood_radius=DEFAULT_NEIGHBORHOOD_RADIUS,
    )
    obs_mean = weighted_mean(obs, weights)
    threshold_mean = weighted_mean(threshold, weights)
    obs_tail_mean = weighted_top_mean(obs, weights, fraction=DEFAULT_TAIL_FRACTION)
    threshold_tail_mean = weighted_top_mean(threshold, weights, fraction=DEFAULT_TAIL_FRACTION)
    obs_event_fraction = weighted_mean((obs >= threshold).astype(np.float32), weights)
    row = {
        "event_id": event["event_id"],
        "region": event["region"],
        "region_label": event["region_label"],
        "variable": variable,
        "event_name": event["event_name"],
        "event_start": str(pd.Timestamp(event["event_start"]).date()),
        "event_end": str(pd.Timestamp(event["event_end"]).date()),
        "init_time": str(pd.Timestamp(sample["init_time"]).date()),
        "valid_time": str(valid_time.date()),
        "target_window_start": str(pd.Timestamp(sample["target_window_start"]).date()),
        "target_window_end": str(pd.Timestamp(sample["target_window_end"]).date()),
        "lead": int(sample["lead"]),
        "lead_label": f"week{int(sample['lead'])}",
        "event_distance_days": int(sample["event_distance_days"]),
        "event_overlap_days": int(sample.get("event_overlap_days", 0)),
        "event_overlap_fraction": float(sample.get("event_overlap_fraction", 0.0)),
        "in_event_window": bool(sample["in_event_window"]),
        "selection_mode": sample.get("selection_mode", "timeseries"),
        "obs_mean": obs_mean,
        "threshold_mean": threshold_mean,
        "obs_tail_mean": obs_tail_mean,
        "threshold_tail_mean": threshold_tail_mean,
        "obs_event_fraction": obs_event_fraction,
        "model_mean": model["mean"],
        "geos_mean": geos["mean"],
        "area_mean_closeness_gain": abs(geos["mean"] - obs_mean) - abs(model["mean"] - obs_mean),
        "model_spread_region_mean": model["spread_region_mean"],
        "geos_spread_region_mean": geos["spread_region_mean"],
        "model_q10_region_mean": model["q10_region_mean"],
        "model_q90_region_mean": model["q90_region_mean"],
        "model_q95_region_mean": model["q95_region_mean"],
        "geos_q10_region_mean": geos["q10_region_mean"],
        "geos_q90_region_mean": geos["q90_region_mean"],
        "geos_q95_region_mean": geos["q95_region_mean"],
        "model_upper_quantile_mean": model["upper_quantile_mean"],
        "geos_upper_quantile_mean": geos["upper_quantile_mean"],
        "upper_quantile_mean_closeness_gain": (
            abs(geos["upper_quantile_mean"] - obs_mean) - abs(model["upper_quantile_mean"] - obs_mean)
        ),
        "model_upper_quantile_tail_mean": model["upper_quantile_tail_mean"],
        "geos_upper_quantile_tail_mean": geos["upper_quantile_tail_mean"],
        "upper_quantile_tail_closeness_gain": (
            abs(geos["upper_quantile_tail_mean"] - obs_tail_mean)
            - abs(model["upper_quantile_tail_mean"] - obs_tail_mean)
        ),
        "model_tail_mean": model["tail_mean"],
        "geos_tail_mean": geos["tail_mean"],
        "model_tail_spread": model["tail_spread"],
        "geos_tail_spread": geos["tail_spread"],
        "model_tail_q10": model["tail_q10"],
        "model_tail_q90": model["tail_q90"],
        "model_tail_q95": model["tail_q95"],
        "geos_tail_q10": geos["tail_q10"],
        "geos_tail_q90": geos["tail_q90"],
        "geos_tail_q95": geos["tail_q95"],
        "tail_closeness_gain": abs(geos["tail_mean"] - obs_tail_mean) - abs(model["tail_mean"] - obs_tail_mean),
        "model_rmse": model["rmse"],
        "geos_rmse": geos["rmse"],
        "rmse_skill_pct": skill_pct(model["rmse"], geos["rmse"]),
        "model_mae": model["mae"],
        "geos_mae": geos["mae"],
        "mae_skill_pct": skill_pct(model["mae"], geos["mae"]),
        "mae_closeness_gain": geos["mae"] - model["mae"],
        "model_bias": model["bias"],
        "geos_bias": geos["bias"],
        "model_crps": model["crps"],
        "geos_crps": geos["crps"],
        "crps_skill_pct": skill_pct(model["crps"], geos["crps"]),
        "model_crps_on_obs_extreme": model["crps_on_obs_extreme"],
        "geos_crps_on_obs_extreme": geos["crps_on_obs_extreme"],
        "crps_on_obs_extreme_skill_pct": skill_pct(model["crps_on_obs_extreme"], geos["crps_on_obs_extreme"]),
        "model_spread_grid": model["spread_grid"],
        "geos_spread_grid": geos["spread_grid"],
        "model_event_probability": model["event_probability"],
        "geos_event_probability": geos["event_probability"],
        "model_event_probability_calibrated": model["event_probability_calibrated"],
        "geos_event_probability_calibrated": geos["event_probability_calibrated"],
        "model_event_probability_neighborhood": model["event_probability_neighborhood"],
        "geos_event_probability_neighborhood": geos["event_probability_neighborhood"],
        "event_probability_neighborhood_diff": (
            model["event_probability_neighborhood"] - geos["event_probability_neighborhood"]
        ),
        "model_event_probability_top_tail": model["event_probability_top_tail"],
        "geos_event_probability_top_tail": geos["event_probability_top_tail"],
        "model_event_probability_calibrated_top_tail": model["event_probability_calibrated_top_tail"],
        "geos_event_probability_calibrated_top_tail": geos["event_probability_calibrated_top_tail"],
        "model_event_probability_neighborhood_top_tail": model["event_probability_neighborhood_top_tail"],
        "geos_event_probability_neighborhood_top_tail": geos["event_probability_neighborhood_top_tail"],
        "event_probability_neighborhood_top_tail_diff": (
            model["event_probability_neighborhood_top_tail"] - geos["event_probability_neighborhood_top_tail"]
        ),
        "model_event_probability_on_obs_extreme": model["event_probability_on_obs_extreme"],
        "geos_event_probability_on_obs_extreme": geos["event_probability_on_obs_extreme"],
        "event_probability_on_obs_extreme_diff": (
            model["event_probability_on_obs_extreme"] - geos["event_probability_on_obs_extreme"]
        ),
        "model_event_probability_on_obs_nonextreme": model["event_probability_on_obs_nonextreme"],
        "geos_event_probability_on_obs_nonextreme": geos["event_probability_on_obs_nonextreme"],
        "event_probability_on_obs_nonextreme_diff": (
            model["event_probability_on_obs_nonextreme"] - geos["event_probability_on_obs_nonextreme"]
        ),
        "model_event_probability_neighborhood_on_obs_extreme": model["event_probability_neighborhood_on_obs_extreme"],
        "geos_event_probability_neighborhood_on_obs_extreme": geos["event_probability_neighborhood_on_obs_extreme"],
        "event_probability_neighborhood_on_obs_extreme_diff": (
            model["event_probability_neighborhood_on_obs_extreme"]
            - geos["event_probability_neighborhood_on_obs_extreme"]
        ),
        "model_event_probability_neighborhood_on_obs_nonextreme": model[
            "event_probability_neighborhood_on_obs_nonextreme"
        ],
        "geos_event_probability_neighborhood_on_obs_nonextreme": geos[
            "event_probability_neighborhood_on_obs_nonextreme"
        ],
        "event_probability_neighborhood_on_obs_nonextreme_diff": (
            model["event_probability_neighborhood_on_obs_nonextreme"]
            - geos["event_probability_neighborhood_on_obs_nonextreme"]
        ),
        "model_cal_event_probability_on_obs_extreme": model["cal_event_probability_on_obs_extreme"],
        "geos_cal_event_probability_on_obs_extreme": geos["cal_event_probability_on_obs_extreme"],
        "cal_event_probability_on_obs_extreme_diff": (
            model["cal_event_probability_on_obs_extreme"] - geos["cal_event_probability_on_obs_extreme"]
        ),
        "model_cal_event_probability_on_obs_nonextreme": model["cal_event_probability_on_obs_nonextreme"],
        "geos_cal_event_probability_on_obs_nonextreme": geos["cal_event_probability_on_obs_nonextreme"],
        "cal_event_probability_on_obs_nonextreme_diff": (
            model["cal_event_probability_on_obs_nonextreme"] - geos["cal_event_probability_on_obs_nonextreme"]
        ),
        "model_event_area_fraction_prob50": model["event_area_fraction_prob50"],
        "geos_event_area_fraction_prob50": geos["event_area_fraction_prob50"],
        "model_cal_event_area_fraction_prob50": model["cal_event_area_fraction_prob50"],
        "geos_cal_event_area_fraction_prob50": geos["cal_event_area_fraction_prob50"],
        "model_neighborhood_event_area_fraction_prob50": model["neighborhood_event_area_fraction_prob50"],
        "geos_neighborhood_event_area_fraction_prob50": geos["neighborhood_event_area_fraction_prob50"],
        "model_bss": model["bss"],
        "geos_bss": geos["bss"],
        "bss_diff": model["bss"] - geos["bss"],
        "model_calibrated_bss": model["calibrated_bss"],
        "geos_calibrated_bss": geos["calibrated_bss"],
        "calibrated_bss_diff": model["calibrated_bss"] - geos["calibrated_bss"],
    }
    maps = {
        "obs": obs,
        "threshold": threshold,
        "obs_event": (obs >= threshold).astype(np.float32),
        "model_mean": model["mean_map"],
        "geos_mean": geos["mean_map"],
        "model_upper_quantile": model["upper_quantile_map"],
        "geos_upper_quantile": geos["upper_quantile_map"],
        "model_crps": model["crps_map"],
        "geos_crps": geos["crps_map"],
        "model_bss": model["bss_map"],
        "geos_bss": geos["bss_map"],
        "model_bss_cal": model["bss_cal_map"],
        "geos_bss_cal": geos["bss_cal_map"],
        "model_prob": model["prob_map"],
        "geos_prob": geos["prob_map"],
        "model_prob_cal": model["prob_cal_map"],
        "geos_prob_cal": geos["prob_cal_map"],
        "model_prob_neighborhood": model["prob_neighborhood_map"],
        "geos_prob_neighborhood": geos["prob_neighborhood_map"],
    }
    return row, maps


def plot_units(row, variable):
    offset = VARIABLES[variable]["offset"]
    out = dict(row)
    for col in [
        "obs_mean",
        "threshold_mean",
        "obs_tail_mean",
        "threshold_tail_mean",
        "model_mean",
        "geos_mean",
        "model_q10_region_mean",
        "model_q90_region_mean",
        "model_q95_region_mean",
        "geos_q10_region_mean",
        "geos_q90_region_mean",
        "geos_q95_region_mean",
        "model_upper_quantile_mean",
        "geos_upper_quantile_mean",
        "model_upper_quantile_tail_mean",
        "geos_upper_quantile_tail_mean",
        "model_tail_mean",
        "geos_tail_mean",
        "model_tail_q10",
        "model_tail_q90",
        "model_tail_q95",
        "geos_tail_q10",
        "geos_tail_q90",
        "geos_tail_q95",
    ]:
        out[col] = out[col] + offset if np.isfinite(out[col]) else out[col]
    return out


def _unique_legend(fig, axes, ncol=4):
    handles = []
    labels = []
    seen = set()
    for ax in np.asarray(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.995), fontsize=7)


def plot_event_timeseries(event, rows, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variable = event["variable"]
    units = VARIABLES[variable]["plot_units"]
    plot_dir = os.path.join(out_dir, "plots", "timeseries")
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return None
    df["valid_time_dt"] = pd.to_datetime(df["valid_time"])
    event_start = pd.Timestamp(event["event_start"])
    event_end = pd.Timestamp(event["event_end"])
    q_label = f"p{int(round(DEFAULT_MAP_QUANTILE * 100))}"
    fig, axes = plt.subplots(len(DEFAULT_LEADS), 5, figsize=(27, 3.8 * len(DEFAULT_LEADS) + 1.0), sharex="col")
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for row_idx, lead in enumerate(DEFAULT_LEADS):
        lead_df = df[df["lead"].eq(lead)].sort_values("valid_time_dt")
        if lead_df.empty:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue
        p = lead_df.apply(lambda row: plot_units(row, variable), axis=1, result_type="expand")
        x = lead_df["valid_time_dt"].values

        ax = axes[row_idx, 0]
        ax.plot(x, p["obs_mean"], color="black", marker="o", label="Obs")
        ax.plot(x, p["model_mean"], color="#1f77b4", marker="o", label="ML mean")
        ax.fill_between(
            x,
            p["model_q10_region_mean"].astype(float),
            p["model_q90_region_mean"].astype(float),
            color="#1f77b4",
            alpha=0.20,
            label="ML p10-p90",
        )
        ax.plot(x, p["geos_mean"], color="#ff7f0e", marker="o", label="GEOS mean")
        ax.fill_between(
            x,
            p["geos_q10_region_mean"].astype(float),
            p["geos_q90_region_mean"].astype(float),
            color="#ff7f0e",
            alpha=0.18,
            label="GEOS p10-p90",
        )
        ax.axvspan(event_start, event_end, color="0.2", alpha=0.10, label="event window")
        ax.set_ylabel(f"{variable.upper()} ({units})")
        ax.set_title(f"lead week {lead}: area mean\nsecondary diagnostic")
        ax.grid(alpha=0.25)

        ax = axes[row_idx, 1]
        ax.plot(x, p["obs_tail_mean"], color="black", marker="o", label="Obs tail")
        ax.plot(x, p["threshold_tail_mean"], color="0.35", linestyle="--", linewidth=1.2, label="obs-clim tail threshold")
        ax.plot(x, p["model_tail_mean"], color="#1f77b4", marker="o", label="ML tail")
        ax.fill_between(
            x,
            p["model_tail_q10"].astype(float),
            p["model_tail_q90"].astype(float),
            color="#1f77b4",
            alpha=0.20,
            label="ML tail p10-p90",
        )
        ax.plot(
            x,
            p["model_upper_quantile_tail_mean"],
            color="#1f77b4",
            linestyle="-.",
            linewidth=1.3,
            label=f"ML {q_label} tail",
        )
        ax.plot(x, p["geos_tail_mean"], color="#ff7f0e", marker="o", label="GEOS tail")
        ax.fill_between(
            x,
            p["geos_tail_q10"].astype(float),
            p["geos_tail_q90"].astype(float),
            color="#ff7f0e",
            alpha=0.18,
            label="GEOS tail p10-p90",
        )
        ax.plot(
            x,
            p["geos_upper_quantile_tail_mean"],
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.3,
            label=f"GEOS {q_label} tail",
        )
        ax.axvspan(event_start, event_end, color="0.2", alpha=0.10)
        ax.set_title(f"lead week {lead}: top {DEFAULT_TAIL_FRACTION:.0%} intensity")
        ax.grid(alpha=0.25)

        ax = axes[row_idx, 2]
        ax.plot(x, p["obs_event_fraction"], color="black", marker="o", label="Obs extreme area fraction")
        ax.plot(
            x,
            p["model_event_probability"],
            color="#1f77b4",
            marker="o",
            label="ML raw event probability",
        )
        ax.plot(
            x,
            p["geos_event_probability"],
            color="#ff7f0e",
            marker="o",
            label="GEOS raw event probability",
        )
        ax.plot(
            x,
            p["model_event_probability_calibrated"],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
            label="ML calibrated event probability",
        )
        ax.plot(
            x,
            p["geos_event_probability_calibrated"],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.2,
            label="GEOS calibrated event probability",
        )
        ax.axvspan(event_start, event_end, color="0.2", alpha=0.10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"lead week {lead}: event-box risk")
        ax.grid(alpha=0.25)

        ax = axes[row_idx, 3]
        ax.plot(
            x,
            p["obs_event_fraction"],
            color="black",
            marker="o",
            label="Obs extreme area fraction",
        )
        ax.plot(
            x,
            p["model_event_probability_on_obs_extreme"],
            color="#1f77b4",
            marker="o",
            label="ML prob on obs-extreme cells",
        )
        ax.plot(
            x,
            p["geos_event_probability_on_obs_extreme"],
            color="#ff7f0e",
            marker="o",
            label="GEOS prob on obs-extreme cells",
        )
        ax.plot(
            x,
            p["model_event_probability_neighborhood_on_obs_extreme"],
            color="#1f77b4",
            linestyle=":",
            linewidth=1.4,
            label=f"ML neighborhood prob r={DEFAULT_NEIGHBORHOOD_RADIUS}",
        )
        ax.plot(
            x,
            p["geos_event_probability_neighborhood_on_obs_extreme"],
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.4,
            label=f"GEOS neighborhood prob r={DEFAULT_NEIGHBORHOOD_RADIUS}",
        )
        ax.axvspan(event_start, event_end, color="0.2", alpha=0.10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"lead week {lead}: risk where obs was extreme")
        ax.grid(alpha=0.25)

        ax = axes[row_idx, 4]
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="tie")
        ax.plot(
            x,
            p["tail_closeness_gain"],
            color="#1f77b4",
            marker="o",
            label="mean-tail closeness",
        )
        ax.plot(
            x,
            p["upper_quantile_tail_closeness_gain"],
            color="#9467bd",
            marker="o",
            label=f"{q_label}-tail closeness",
        )
        ax.axvspan(event_start, event_end, color="0.2", alpha=0.10)
        ax.set_title(f"lead week {lead}: tail closeness vs obs\npositive = ML closer")
        ax.grid(alpha=0.25)
    _unique_legend(fig, axes, ncol=5)
    fig.suptitle(f"{event['region_label']} | {event['event_name']} | {variable.upper()}", y=0.83)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.75])
    out_path = os.path.join(plot_dir, f"{event['event_id']}_timeseries.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def shade_event_overlap_by_lead(ax, df):
    for _, row in df.iterrows():
        overlap = float(row.get("event_overlap_fraction", 0.0))
        if overlap <= 0:
            continue
        lead = float(row["lead"])
        ax.axvspan(lead - 0.38, lead + 0.38, color="0.2", alpha=min(0.25, 0.05 + 0.25 * overlap))


def plot_fixed_init_progression(event, rows, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variable = event["variable"]
    units = VARIABLES[variable]["plot_units"]
    plot_dir = os.path.join(out_dir, "plots", "fixed_init_progression")
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return []
    out_paths = []
    q_label = f"p{int(round(DEFAULT_MAP_QUANTILE * 100))}"
    for init_time, group in df.groupby("init_time", sort=True):
        group = group.sort_values("lead")
        p = group.apply(lambda row: plot_units(row, variable), axis=1, result_type="expand")
        x = p["lead"].astype(float).values
        fig, axes = plt.subplots(1, 5, figsize=(26, 4.6), sharex=True)

        ax = axes[0]
        shade_event_overlap_by_lead(ax, p)
        ax.plot(x, p["obs_mean"], color="black", marker="o", label="Obs")
        ax.plot(x, p["model_mean"], color="#1f77b4", marker="o", label="ML mean")
        ax.fill_between(
            x,
            p["model_q10_region_mean"].astype(float),
            p["model_q90_region_mean"].astype(float),
            color="#1f77b4",
            alpha=0.20,
            label="ML p10-p90",
        )
        ax.plot(x, p["geos_mean"], color="#ff7f0e", marker="o", label="GEOS mean")
        ax.fill_between(
            x,
            p["geos_q10_region_mean"].astype(float),
            p["geos_q90_region_mean"].astype(float),
            color="#ff7f0e",
            alpha=0.18,
            label="GEOS p10-p90",
        )
        ax.set_ylabel(f"{variable.upper()} ({units})")
        ax.set_title("area mean\nsecondary diagnostic")
        ax.grid(alpha=0.25)

        ax = axes[1]
        shade_event_overlap_by_lead(ax, p)
        ax.plot(x, p["obs_tail_mean"], color="black", marker="o", label="Obs tail")
        ax.plot(x, p["threshold_tail_mean"], color="0.35", linestyle="--", linewidth=1.2, label="obs-clim tail threshold")
        ax.plot(x, p["model_tail_mean"], color="#1f77b4", marker="o", label="ML tail")
        ax.fill_between(
            x,
            p["model_tail_q10"].astype(float),
            p["model_tail_q90"].astype(float),
            color="#1f77b4",
            alpha=0.20,
            label="ML tail p10-p90",
        )
        ax.plot(
            x,
            p["model_upper_quantile_tail_mean"],
            color="#1f77b4",
            linestyle="-.",
            linewidth=1.3,
            label=f"ML {q_label} tail",
        )
        ax.plot(x, p["geos_tail_mean"], color="#ff7f0e", marker="o", label="GEOS tail")
        ax.fill_between(
            x,
            p["geos_tail_q10"].astype(float),
            p["geos_tail_q90"].astype(float),
            color="#ff7f0e",
            alpha=0.18,
            label="GEOS tail p10-p90",
        )
        ax.plot(
            x,
            p["geos_upper_quantile_tail_mean"],
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.3,
            label=f"GEOS {q_label} tail",
        )
        ax.set_title(f"top {DEFAULT_TAIL_FRACTION:.0%} intensity")
        ax.grid(alpha=0.25)

        ax = axes[2]
        shade_event_overlap_by_lead(ax, p)
        ax.plot(x, p["obs_event_fraction"], color="black", marker="o", label="Obs extreme area fraction")
        ax.plot(x, p["model_event_probability"], color="#1f77b4", marker="o", label="ML raw event prob")
        ax.plot(x, p["geos_event_probability"], color="#ff7f0e", marker="o", label="GEOS raw event prob")
        ax.plot(
            x,
            p["model_event_probability_calibrated"],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
            label="ML calibrated event prob",
        )
        ax.plot(
            x,
            p["geos_event_probability_calibrated"],
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.2,
            label="GEOS calibrated event prob",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("event-box risk")
        ax.grid(alpha=0.25)

        ax = axes[3]
        shade_event_overlap_by_lead(ax, p)
        ax.plot(x, p["obs_event_fraction"], color="black", marker="o", label="Obs extreme area fraction")
        ax.plot(
            x,
            p["model_event_probability_on_obs_extreme"],
            color="#1f77b4",
            marker="o",
            label="ML prob on obs-extreme cells",
        )
        ax.plot(
            x,
            p["geos_event_probability_on_obs_extreme"],
            color="#ff7f0e",
            marker="o",
            label="GEOS prob on obs-extreme cells",
        )
        ax.plot(
            x,
            p["model_event_probability_neighborhood_on_obs_extreme"],
            color="#1f77b4",
            linestyle=":",
            linewidth=1.4,
            label=f"ML neighborhood prob r={DEFAULT_NEIGHBORHOOD_RADIUS}",
        )
        ax.plot(
            x,
            p["geos_event_probability_neighborhood_on_obs_extreme"],
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.4,
            label=f"GEOS neighborhood prob r={DEFAULT_NEIGHBORHOOD_RADIUS}",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("risk where obs was extreme")
        ax.grid(alpha=0.25)

        ax = axes[4]
        shade_event_overlap_by_lead(ax, p)
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="tie")
        ax.plot(x, p["tail_closeness_gain"], color="#1f77b4", marker="o", label="mean-tail closeness")
        ax.plot(
            x,
            p["upper_quantile_tail_closeness_gain"],
            color="#9467bd",
            marker="o",
            label=f"{q_label}-tail closeness",
        )
        ax.set_title("tail closeness vs obs\npositive = ML closer")
        ax.grid(alpha=0.25)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels([f"week{int(v)}" for v in x])
            ax.set_xlabel("lead")

        handles = []
        labels = []
        seen = set()
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label and label not in seen:
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=6, bbox_to_anchor=(0.5, 0.995))
        valid_windows = ", ".join(
            f"w{int(row.lead)}:{row.target_window_start}..{row.target_window_end}"
            for row in group.itertuples(index=False)
        )
        fig.suptitle(
            f"{event['region_label']} | {event['event_name']} | {variable.upper()} | "
            f"fixed init {init_time} | target windows {valid_windows}",
            y=0.85,
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.74])
        init_label = pd.Timestamp(init_time).strftime("%Y%m%d")
        out_path = os.path.join(plot_dir, f"{event['event_id']}_init{init_label}_lead_progression.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def prepare_region_plot(lons, lats, field, bbox, mask):
    lon180 = lon_to_180(lons)
    order = np.argsort(lon180)
    lon_sorted = lon180[order]
    field_sorted = np.asarray(field)[:, order]
    if mask is not None:
        mask_sorted = np.asarray(mask)[:, order]
        field_sorted = np.where(mask_sorted, field_sorted, np.nan)
    lon_min, lon_max, lat_min, lat_max = [float(x) for x in bbox]
    lon_keep = (lon_sorted >= lon_min - 2.0) & (lon_sorted <= lon_max + 2.0)
    lat_values = np.asarray(lats, dtype=np.float64)
    lat_keep = (lat_values >= lat_min - 2.0) & (lat_values <= lat_max + 2.0)
    return lon_sorted[lon_keep], lat_values[lat_keep], field_sorted[np.ix_(lat_keep, lon_keep)]


def plot_event_spatial(event, sample_row, maps, lons, lats, mask, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots", "spatial_maps")
    os.makedirs(plot_dir, exist_ok=True)
    variable = event["variable"]
    offset = VARIABLES[variable]["offset"]
    units = VARIABLES[variable]["plot_units"]
    intensity_cmap = "coolwarm" if variable == "t2m" else "viridis"
    def masked_for_limits(field):
        return np.where(mask, field, np.nan)

    event_mask = mask & (maps["obs_event"] > 0)

    def observed_event_only(field):
        return np.where(event_mask, field, np.nan)

    obs_plot = maps["obs"] + offset
    threshold_plot = maps["threshold"] + offset
    geos_plot = maps["geos_mean"] + offset
    model_plot = maps["model_mean"] + offset
    geos_upper_plot = maps["geos_upper_quantile"] + offset
    model_upper_plot = maps["model_upper_quantile"] + offset
    model_minus_geos = maps["model_mean"] - maps["geos_mean"]
    upper_minus_geos = maps["model_upper_quantile"] - maps["geos_upper_quantile"]
    geos_abs_error = np.abs(maps["geos_mean"] - maps["obs"])
    model_abs_error = np.abs(maps["model_mean"] - maps["obs"])
    closeness_gain = geos_abs_error - model_abs_error
    geos_upper_abs_error = np.abs(maps["geos_upper_quantile"] - maps["obs"])
    model_upper_abs_error = np.abs(maps["model_upper_quantile"] - maps["obs"])
    upper_closeness_gain = geos_upper_abs_error - model_upper_abs_error
    crps_skill = np.where(
        np.abs(maps["geos_crps"]) > 1e-12,
        100.0 * (1.0 - maps["model_crps"] / maps["geos_crps"]),
        np.nan,
    )
    raw_prob_diff = maps["model_prob"] - maps["geos_prob"]
    cal_prob_diff = maps["model_prob_cal"] - maps["geos_prob_cal"]
    neighborhood_prob_diff = maps["model_prob_neighborhood"] - maps["geos_prob_neighborhood"]
    bss_diff = maps["model_bss"] - maps["geos_bss"]
    cal_bss_diff = maps["model_bss_cal"] - maps["geos_bss_cal"]
    q_label = f"p{int(round(DEFAULT_MAP_QUANTILE * 100))}"
    intensity_vmin, intensity_vmax = robust_limits(
        [masked_for_limits(obs_plot), masked_for_limits(threshold_plot), masked_for_limits(geos_plot), masked_for_limits(model_plot)]
    )
    upper_vmin, upper_vmax = robust_limits(
        [
            masked_for_limits(obs_plot),
            masked_for_limits(threshold_plot),
            masked_for_limits(geos_upper_plot),
            masked_for_limits(model_upper_plot),
        ]
    )
    error_vmin, error_vmax = robust_limits(
        [masked_for_limits(geos_abs_error), masked_for_limits(model_abs_error)], lower=0, upper=95, fallback=(0.0, 1.0)
    )
    diff_vmin, diff_vmax = robust_limits([masked_for_limits(model_minus_geos)], center_zero=True)
    upper_diff_vmin, upper_diff_vmax = robust_limits([masked_for_limits(upper_minus_geos)], center_zero=True)
    closeness_vmin, closeness_vmax = robust_limits([masked_for_limits(closeness_gain)], center_zero=True)
    upper_closeness_vmin, upper_closeness_vmax = robust_limits(
        [masked_for_limits(upper_closeness_gain)],
        center_zero=True,
    )
    crps_vmin, crps_vmax = robust_limits([masked_for_limits(crps_skill)], center_zero=True)
    prob_diff_vmin, prob_diff_vmax = robust_limits(
        [
            masked_for_limits(raw_prob_diff),
            masked_for_limits(cal_prob_diff),
            masked_for_limits(neighborhood_prob_diff),
        ],
        center_zero=True,
        fallback=(-1.0, 1.0),
    )
    raw_prob_diff_vmin, raw_prob_diff_vmax = prob_diff_vmin, prob_diff_vmax
    cal_prob_diff_vmin, cal_prob_diff_vmax = prob_diff_vmin, prob_diff_vmax
    neighborhood_prob_diff_vmin, neighborhood_prob_diff_vmax = prob_diff_vmin, prob_diff_vmax
    bss_diff_vmin, bss_diff_vmax = robust_limits(
        [masked_for_limits(bss_diff), masked_for_limits(cal_bss_diff)],
        center_zero=True,
        fallback=(-1.0, 1.0),
    )
    focus_prob_diff_vmin, focus_prob_diff_vmax = robust_limits(
        [
            observed_event_only(raw_prob_diff),
            observed_event_only(cal_prob_diff),
            observed_event_only(neighborhood_prob_diff),
        ],
        center_zero=True,
        fallback=(-1.0, 1.0),
    )
    focus_bss_vmin, focus_bss_vmax = robust_limits(
        [observed_event_only(bss_diff), observed_event_only(cal_bss_diff)],
        center_zero=True,
        fallback=(-1.0, 1.0),
    )
    def save_spatial_panel_grid(panels, nrows, ncols, figsize, suffix, title_extra, show_panel_titles=True):
        fig, axes = make_map_subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
        from evaluate_matrix_suite_flow_finalv1_global import MAP_CONTEXT

        for ax, (title, field, cmap, center_zero, fixed_vmin, fixed_vmax) in zip(axes.ravel(), panels):
            plot_lons, plot_lats, plot_field = prepare_region_plot(lons, lats, field, event["bbox"], mask)
            finite = plot_field[np.isfinite(plot_field)]
            if fixed_vmin is not None and fixed_vmax is not None:
                vmin, vmax = fixed_vmin, fixed_vmax
            elif finite.size:
                if center_zero:
                    vmax = max(float(np.nanpercentile(np.abs(finite), 95)), 1e-6)
                    vmin = -vmax
                else:
                    vmin, vmax = np.nanpercentile(finite, [5, 95])
            else:
                vmin, vmax = (-1, 1) if center_zero else (0, 1)

            kwargs = {}
            if MAP_CONTEXT["enabled"]:
                kwargs["transform"] = MAP_CONTEXT["data_crs"]
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if center_zero else None
            levels = np.linspace(vmin, vmax, 21)
            mesh = ax.contourf(
                plot_lons,
                plot_lats,
                plot_field,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="both",
                **kwargs,
            )
            add_map_overlays(ax, plot_lons, plot_lats)
            if show_panel_titles:
                ax.set_title(title, fontsize=9)
            ax.set_xlabel("lon", fontsize=8)
            ax.set_ylabel("lat", fontsize=8)
            fig.colorbar(mesh, ax=ax, shrink=0.75)
        for ax in axes.ravel()[len(panels) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"{event['region_label']} | {event['event_name']} | {variable.upper()} | {title_extra} | "
            f"init {sample_row['init_time']} valid {sample_row['valid_time']} lead week {sample_row['lead']} | {units}",
            fontsize=12,
        )
        out_path = os.path.join(plot_dir, f"{event['event_id']}_lead{sample_row['lead']}_{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    risk_panels = [
        ("Observed", obs_plot, intensity_cmap, False, intensity_vmin, intensity_vmax),
        ("Obs clim threshold", threshold_plot, intensity_cmap, False, intensity_vmin, intensity_vmax),
        ("Observed extreme mask", maps["obs_event"], "Greys", False, 0.0, 1.0),
        (f"GEOS {q_label}", geos_upper_plot, intensity_cmap, False, upper_vmin, upper_vmax),
        (f"ML {q_label}", model_upper_plot, intensity_cmap, False, upper_vmin, upper_vmax),
        (f"{q_label} closeness gain\nblue = ML closer", upper_closeness_gain, "RdBu", True, upper_closeness_vmin, upper_closeness_vmax),
        ("GEOS raw event prob", maps["geos_prob"], "viridis", False, 0.0, 1.0),
        ("ML raw event prob", maps["model_prob"], "viridis", False, 0.0, 1.0),
        ("Raw event prob ML-GEOS", raw_prob_diff, "RdBu", True, raw_prob_diff_vmin, raw_prob_diff_vmax),
        (f"GEOS neighborhood prob\nr={DEFAULT_NEIGHBORHOOD_RADIUS}", maps["geos_prob_neighborhood"], "viridis", False, 0.0, 1.0),
        (f"ML neighborhood prob\nr={DEFAULT_NEIGHBORHOOD_RADIUS}", maps["model_prob_neighborhood"], "viridis", False, 0.0, 1.0),
        ("Neighborhood prob ML-GEOS", neighborhood_prob_diff, "RdBu", True, neighborhood_prob_diff_vmin, neighborhood_prob_diff_vmax),
    ]
    verification_panels = [
        ("GEOS mean\nsecondary", geos_plot, intensity_cmap, False, intensity_vmin, intensity_vmax),
        ("ML mean\nsecondary", model_plot, intensity_cmap, False, intensity_vmin, intensity_vmax),
        ("ML - GEOS mean", model_minus_geos, "RdBu", True, diff_vmin, diff_vmax),
        (f"ML - GEOS {q_label}", upper_minus_geos, "RdBu", True, upper_diff_vmin, upper_diff_vmax),
        ("|GEOS mean - Obs|", geos_abs_error, "magma", False, error_vmin, error_vmax),
        ("|ML mean - Obs|", model_abs_error, "magma", False, error_vmin, error_vmax),
        ("Mean closeness gain\nblue = ML closer", closeness_gain, "RdBu", True, closeness_vmin, closeness_vmax),
        ("CRPS skill %", crps_skill, "RdBu", True, crps_vmin, crps_vmax),
        ("BSS ML-GEOS", bss_diff, "RdBu", True, bss_diff_vmin, bss_diff_vmax),
        ("GEOS cal event prob", maps["geos_prob_cal"], "viridis", False, 0.0, 1.0),
        ("ML cal event prob", maps["model_prob_cal"], "viridis", False, 0.0, 1.0),
        ("Cal event prob ML-GEOS", cal_prob_diff, "RdBu", True, cal_prob_diff_vmin, cal_prob_diff_vmax),
        ("Cal BSS ML-GEOS", cal_bss_diff, "RdBu", True, bss_diff_vmin, bss_diff_vmax),
    ]
    focus_panels = [
        ("Observed", obs_plot, intensity_cmap, False, intensity_vmin, intensity_vmax),
        ("GEOS event prob\non observed extremes", observed_event_only(maps["geos_prob"]), "viridis", False, 0.0, 1.0),
        ("ML event prob\non observed extremes", observed_event_only(maps["model_prob"]), "viridis", False, 0.0, 1.0),
        ("Raw probability gain\non observed extremes", observed_event_only(raw_prob_diff), "RdBu", True, focus_prob_diff_vmin, focus_prob_diff_vmax),
        ("Cal probability gain\non observed extremes", observed_event_only(cal_prob_diff), "RdBu", True, focus_prob_diff_vmin, focus_prob_diff_vmax),
        (f"Neighborhood gain\nr={DEFAULT_NEIGHBORHOOD_RADIUS}", observed_event_only(neighborhood_prob_diff), "RdBu", True, focus_prob_diff_vmin, focus_prob_diff_vmax),
        ("BSS gain\non observed extremes", observed_event_only(bss_diff), "RdBu", True, focus_bss_vmin, focus_bss_vmax),
        ("Cal BSS gain\non observed extremes", observed_event_only(cal_bss_diff), "RdBu", True, focus_bss_vmin, focus_bss_vmax),
    ]
    # Save raw spatial data to NetCDF for direct plotting in the paper figures script
    nc_path = os.path.join(out_dir, "plots", "spatial_maps", f"{event['event_id']}_lead{sample_row['lead']}_spatial_data.nc")
    try:
        import xarray as xr
        ds_vars = {}
        for k, field in maps.items():
            plot_lons, plot_lats, plot_field = prepare_region_plot(lons, lats, field, event["bbox"], None)
            ds_vars[k] = (["lat", "lon"], plot_field)
        plot_lons, plot_lats, obs_plot_val = prepare_region_plot(lons, lats, obs_plot, event["bbox"], None)
        ds_vars["obs_plot"] = (["lat", "lon"], obs_plot_val)
        plot_lons, plot_lats, geos_plot_val = prepare_region_plot(lons, lats, geos_plot, event["bbox"], None)
        ds_vars["geos_plot"] = (["lat", "lon"], geos_plot_val)
        plot_lons, plot_lats, model_plot_val = prepare_region_plot(lons, lats, model_plot, event["bbox"], None)
        ds_vars["model_plot"] = (["lat", "lon"], model_plot_val)
        plot_lons, plot_lats, closeness_gain_val = prepare_region_plot(lons, lats, closeness_gain, event["bbox"], None)
        ds_vars["closeness_gain"] = (["lat", "lon"], closeness_gain_val)
        plot_lons, plot_lats, bss_diff_val = prepare_region_plot(lons, lats, bss_diff, event["bbox"], None)
        ds_vars["bss_diff"] = (["lat", "lon"], bss_diff_val)
        plot_lons, plot_lats, cal_bss_diff_val = prepare_region_plot(lons, lats, cal_bss_diff, event["bbox"], None)
        ds_vars["cal_bss_diff"] = (["lat", "lon"], cal_bss_diff_val)

        ds = xr.Dataset(
            data_vars=ds_vars,
            coords={
                "lat": (["lat"], plot_lats),
                "lon": (["lon"], plot_lons),
            }
        )
        ds.to_netcdf(nc_path)
        print(f"💾 Wrote raw spatial NetCDF: {nc_path}")
    except Exception as exc:
        print(f"⚠️ Could not write NetCDF: {exc}")

    return {
        "spatial_event_focus": save_spatial_panel_grid(
            focus_panels,
            2,
            4,
            (19, 8),
            "spatial_event_focus",
            "paper event-focus diagnostics",
            show_panel_titles=False,
        ),
        "spatial_risk": save_spatial_panel_grid(
            risk_panels,
            3,
            4,
            (19, 11),
            "spatial_risk",
            "extreme-risk diagnostics",
        ),
        "spatial_verification": save_spatial_panel_grid(
            verification_panels,
            4,
            4,
            (19, 13),
            "spatial_verification",
            "supporting verification",
        ),
    }


def plot_event_matrix(event_metrics, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots", "event_matrices")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = ["crps_skill_pct", "rmse_skill_pct", "calibrated_bss_diff"]
    for variable in sorted(event_metrics["variable"].unique()):
        data = event_metrics[event_metrics["variable"].eq(variable)].copy()
        event_order = data["event_id"].drop_duplicates().tolist()
        labels = data.drop_duplicates("event_id").set_index("event_id")["region"].to_dict()
        for metric in metrics:
            matrix = np.full((len(event_order), len(DEFAULT_LEADS)), np.nan, dtype=np.float64)
            for i, event_id in enumerate(event_order):
                for j, lead in enumerate(DEFAULT_LEADS):
                    match = data[(data["event_id"] == event_id) & (data["lead"] == lead)]
                    if not match.empty:
                        matrix[i, j] = float(match.iloc[0][metric])
            finite = matrix[np.isfinite(matrix)]
            vmax = max(float(np.nanpercentile(np.abs(finite), 95)), 1e-6) if finite.size else 1.0
            fig, ax = plt.subplots(figsize=(6, max(4, 0.33 * len(event_order))))
            mesh = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(DEFAULT_LEADS)))
            ax.set_xticklabels([f"week{lead}" for lead in DEFAULT_LEADS])
            ax.set_yticks(np.arange(len(event_order)))
            ax.set_yticklabels([labels[event_id] for event_id in event_order], fontsize=7)
            ax.set_title(f"{variable.upper()} event matrix | {metric}")
            fig.colorbar(mesh, ax=ax, shrink=0.85)
            fig.tight_layout()
            out_path = os.path.join(plot_dir, f"{variable}_{metric}_event_matrix.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    leads = parse_list(args.leads, int)
    progression_leads = parse_list(args.progression_leads, int)
    global DEFAULT_LEADS
    DEFAULT_LEADS = leads
    if not (0.0 < float(args.tail_fraction) <= 1.0):
        raise ValueError("--tail_fraction must be >0 and <=1")
    global DEFAULT_TAIL_FRACTION
    DEFAULT_TAIL_FRACTION = float(args.tail_fraction)
    if not (0.0 < float(args.map_quantile) < 1.0):
        raise ValueError("--map_quantile must be >0 and <1")
    global DEFAULT_MAP_QUANTILE
    DEFAULT_MAP_QUANTILE = float(args.map_quantile)
    if int(args.neighborhood_radius) < 0:
        raise ValueError("--neighborhood_radius must be >=0")
    global DEFAULT_NEIGHBORHOOD_RADIUS
    DEFAULT_NEIGHBORHOOD_RADIUS = int(args.neighborhood_radius)
    regions = parse_list(args.regions)
    if regions == ["all"]:
        regions = ["all"]
    variables = parse_list(args.variables)
    bad_variables = [variable for variable in variables if variable not in VARIABLES]
    if bad_variables:
        raise ValueError(f"Unknown variables: {bad_variables}")

    catalog = normalize_catalog(load_event_catalog(args.event_catalog), regions, variables, args.start_year, args.end_year)
    catalog_path = os.path.join(args.out_dir, "event_catalog_used.csv")
    catalog.to_csv(catalog_path, index=False)
    print(f"✅ Wrote event catalog used: {catalog_path}")

    lats, lons = open_forecast_grid(args.forecast_dir, args.start_year, args.end_year)
    land_mask, land_source = load_land_mask(args.land_mask_file, (len(lats), len(lons)))
    thresholds, obs_clim, threshold_lats, threshold_lons = load_thresholds_from_file(args.threshold_file, variables, args)
    if not (np.allclose(lats, threshold_lats) and np.allclose(lons, threshold_lons)):
        raise ValueError("Threshold grid does not match forecast grid.")
    calibrators = load_calibration_params(args.calibration_params)

    if args.make_plots:
        map_args = argparse.Namespace(map_features=args.map_features, county_boundaries=args.county_boundaries)
        configure_map_context(map_args)

    timeseries_rows = []
    event_metric_rows = []
    progression_rows = []
    plot_records = []
    for _, event_row in catalog.iterrows():
        event = event_row.to_dict()
        event["bbox"] = parse_bbox(event["bbox"])
        mask = bbox_mask(lons, lats, event["bbox"]) & land_mask
        if not mask.any():
            print(f"⚠️ {event['event_id']}: empty land mask; skipping.")
            continue
        weights = region_weights(lats, mask, args.regional_weighting)
        samples = find_candidate_samples(
            args.forecast_dir,
            event,
            leads,
            args.timeseries_window_days,
            args.event_tolerance_days,
        )
        if not samples:
            print(f"⚠️ {event['event_id']}: no lead {leads} samples found near event.")
            continue
        selected = choose_event_samples(samples, leads)
        selected_keys = {
            (sample["zarr_path"], sample["init_idx"], sample["lead_idx"])
            for sample in selected
        }
        sample_rows = []
        selected_maps = {}
        for sample in sorted(samples, key=lambda item: (item["lead"], item["valid_time"])):
            row, maps = evaluate_sample(sample, event, thresholds, obs_clim, calibrators, weights)
            is_selected = (sample["zarr_path"], sample["init_idx"], sample["lead_idx"]) in selected_keys
            row["selected_for_event_metrics"] = bool(is_selected)
            sample_rows.append(row)
            timeseries_rows.append(row)
            if is_selected:
                row["selection_mode"] = next(
                    selected_sample["selection_mode"]
                    for selected_sample in selected
                    if (selected_sample["zarr_path"], selected_sample["init_idx"], selected_sample["lead_idx"])
                    == (sample["zarr_path"], sample["init_idx"], sample["lead_idx"])
                )
                event_metric_rows.append(row)
                selected_maps[int(row["lead"])] = (row, maps)
        print(f"✅ {event['event_id']}: {len(sample_rows)} time-series samples, {len(selected_maps)} selected event samples")
        if args.make_plots:
            ts_path = plot_event_timeseries(event, sample_rows, args.out_dir)
            if ts_path:
                plot_records.append({"event_id": event["event_id"], "plot_type": "timeseries", "path": ts_path})

        fixed_samples = fixed_init_progression_samples(selected, event, progression_leads)
        fixed_rows = []
        for sample in fixed_samples:
            row, _ = evaluate_sample(sample, event, thresholds, obs_clim, calibrators, weights)
            row["selected_for_event_metrics"] = False
            row["selection_mode"] = sample.get("selection_mode", "fixed_init_progression")
            fixed_rows.append(row)
            progression_rows.append(row)
        if args.make_plots and fixed_rows:
            for path in plot_fixed_init_progression(event, fixed_rows, args.out_dir):
                plot_records.append({"event_id": event["event_id"], "plot_type": "fixed_init_progression", "path": path})
        if fixed_rows:
            print(f"   ↳ fixed-init progression rows: {len(fixed_rows)} across {len(set(row['init_time'] for row in fixed_rows))} init(s)")

        if args.make_plots:
            for lead, (row, maps) in selected_maps.items():
                map_paths = plot_event_spatial(event, row, maps, lons, lats, mask, args.out_dir)
                for plot_type, map_path in map_paths.items():
                    plot_records.append({"event_id": event["event_id"], "lead": lead, "plot_type": plot_type, "path": map_path})

    timeseries = pd.DataFrame(timeseries_rows)
    event_metrics = pd.DataFrame(event_metric_rows)
    progression = pd.DataFrame(progression_rows)
    timeseries_path = os.path.join(args.out_dir, "event_timeseries_metrics.csv")
    event_metrics_path = os.path.join(args.out_dir, "event_selected_lead_metrics.csv")
    progression_path = os.path.join(args.out_dir, "event_fixed_init_progression_metrics.csv")
    timeseries.to_csv(timeseries_path, index=False, float_format="%.6f")
    event_metrics.to_csv(event_metrics_path, index=False, float_format="%.6f")
    progression.to_csv(progression_path, index=False, float_format="%.6f")
    print(f"✅ Wrote event time-series metrics: {timeseries_path}")
    print(f"✅ Wrote selected event-lead metrics: {event_metrics_path}")
    print(f"✅ Wrote fixed-init progression metrics: {progression_path}")

    if not event_metrics.empty:
        overall_rows = []
        for key, group in event_metrics.groupby(["region", "region_label", "variable"]):
            region, region_label, variable = key
            overall_rows.append(
                {
                    "region": region,
                    "region_label": region_label,
                    "variable": variable,
                    "n_selected_leads": int(len(group)),
                    "mean_crps_skill_pct": float(np.nanmean(group["crps_skill_pct"])),
                    "mean_rmse_skill_pct": float(np.nanmean(group["rmse_skill_pct"])),
                    "mean_calibrated_bss_diff": float(np.nanmean(group["calibrated_bss_diff"])),
                    "all_leads_crps_positive": bool((group["crps_skill_pct"] > 0).all()),
                    "all_leads_rmse_positive": bool((group["rmse_skill_pct"] > 0).all()),
                }
            )
        overall = pd.DataFrame(overall_rows)
        overall_path = os.path.join(args.out_dir, "event_overall_by_region_variable.csv")
        overall.to_csv(overall_path, index=False, float_format="%.6f")
        print(f"✅ Wrote event overall table: {overall_path}")
        if args.make_plots:
            plot_event_matrix(event_metrics, args.out_dir)

    plot_index_path = os.path.join(args.out_dir, "event_plot_index.csv")
    pd.DataFrame(plot_records).to_csv(plot_index_path, index=False)
    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "threshold_file": os.path.abspath(args.threshold_file),
        "calibration_params": os.path.abspath(args.calibration_params) if os.path.exists(args.calibration_params) else None,
        "land_mask_file": land_source,
        "leads": leads,
        "progression_leads": progression_leads,
        "tail_fraction": float(DEFAULT_TAIL_FRACTION),
        "map_quantile": float(DEFAULT_MAP_QUANTILE),
        "neighborhood_radius": int(DEFAULT_NEIGHBORHOOD_RADIUS),
        "regional_weighting": args.regional_weighting,
        "timeseries_window_days": args.timeseries_window_days,
        "event_tolerance_days": args.event_tolerance_days,
        "event_count": int(len(catalog)),
        "note": (
            "Forecast data are weekly lead targets. Each valid date is treated as a 7-day target window ending "
            "on that date for event-overlap selection. Time series show weekly valid times around each event. "
            "Large-ensemble event diagnostics emphasize exceedance probabilities, upper quantiles, neighborhood "
            "probabilities, and top-tail intensity; ensemble means are retained as secondary references. Spread "
            "bands are p10-p90 of regional ensemble-mean or top-tail values. Spatial maps are split into compact "
            "extreme-risk and supporting-verification figures to avoid mixing the main forecast-risk story with "
            "secondary mean/error diagnostics."
        ),
    }
    metadata_path = os.path.join(args.out_dir, "event_catalog_eval_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
