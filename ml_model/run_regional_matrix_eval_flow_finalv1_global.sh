#!/bin/bash
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/11353/afahad/geossub/geos_subc}"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

if [ -d "$CONDA_ENV_PATH" ]; then
    source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

MATRIX_SPATIAL_FILE="${REGIONAL_MATRIX_SPATIAL_FILE:-ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked/matrix_spatial_metrics.nc}"
METADATA_FILE="${REGIONAL_MATRIX_METADATA_FILE:-ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked/matrix_eval_metadata.json}"
OUT_DIR="${REGIONAL_MATRIX_OUT_DIR:-ml_output_flow_finalv1_global_noisectx_t2mres/regional_matrix_eval_global_2021_2023_land_obsclim}"
LAND_MASK_FILE="${REGIONAL_LAND_MASK_FILE:-ml_model/land_ocean_mask_v6.pt}"
REGIONS="${REGIONAL_REGIONS:-all}"
VARIABLES="${REGIONAL_VARIABLES:-pr,t2m}"
SUBSETS="${REGIONAL_SUBSETS:-all_data,extreme_events}"
MAKE_MAPS="${REGIONAL_MAKE_MAPS:-1}"
MAP_FEATURES="${REGIONAL_MAP_FEATURES:-auto}"
COUNTY_BOUNDARIES="${REGIONAL_COUNTY_BOUNDARIES:-off}"
MASK_SOURCE="${REGIONAL_MASK_SOURCE:-auto}"
PLOT_METRICS="${REGIONAL_PLOT_METRICS:-crps_skill_pct,rmse_skill_pct,calibrated_bss_diff}"
PLOT_GROUP_TYPE="${REGIONAL_PLOT_GROUP_TYPE:-valid_season_lead}"
OVERWRITE="${REGIONAL_OVERWRITE:-1}"

EXTRA_ARGS=()
if [ "$MAKE_MAPS" = "1" ] || [ "$MAKE_MAPS" = "true" ] || [ "$MAKE_MAPS" = "TRUE" ]; then
    EXTRA_ARGS+=(--make_maps)
fi
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "TRUE" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "🚀 Regional matrix evaluation started at $(date) on $(hostname)"
echo "🎯 Matrix spatial file: $MATRIX_SPATIAL_FILE"
echo "🎯 Regions: $REGIONS"
echo "🎯 Variables: $VARIABLES subsets=$SUBSETS"
echo "🎯 Output: $OUT_DIR"
echo "🎯 Maps: $MAKE_MAPS map_features=$MAP_FEATURES plot_metrics=$PLOT_METRICS"

python ml_model/evaluate_regional_matrix_flow_finalv1_global.py \
    --matrix_spatial_file "$MATRIX_SPATIAL_FILE" \
    --metadata_file "$METADATA_FILE" \
    --out_dir "$OUT_DIR" \
    --land_mask_file "$LAND_MASK_FILE" \
    --regions "$REGIONS" \
    --variables "$VARIABLES" \
    --subsets "$SUBSETS" \
    --mask_source "$MASK_SOURCE" \
    --plot_metrics "$PLOT_METRICS" \
    --plot_group_type "$PLOT_GROUP_TYPE" \
    --map_features "$MAP_FEATURES" \
    --county_boundaries "$COUNTY_BOUNDARIES" \
    "${EXTRA_ARGS[@]}"

echo "🏁 Regional matrix evaluation finished at $(date)"
