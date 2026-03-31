#!/bin/bash
#SBATCH -J gen_mv1_base
#SBATCH -o ml_output_flowmulti/gen_multiv1_baseline_1999_2019_%j.log
#SBATCH -e ml_output_flowmulti/gen_multiv1_baseline_1999_2019_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Multi-v1 ML hindcast generation job started at $(date) on $(hostname)"

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest code..."
git pull --no-rebase origin flow_multi

mkdir -p ml_output_flowmulti
mkdir -p dataprocess/gen_multiv1_hindcast_1999_2019

CONFIG_PATH="ml_model/config_flow_multiv1.yaml"
START_YEAR=1999
END_YEAR=2019
MAX_NEW_INITS=540
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"

echo "🎯 Generating 4-member ML hindcast ensembles for ${START_YEAR}-${END_YEAR}"
echo "   Session limit: ${MAX_NEW_INITS} new init dates per job"
accelerate launch --num_processes 1 --mixed_precision fp16 \
    ml_model/generate_multiv1_baseline_1999_2019.py \
    --config "$CONFIG_PATH" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --num_ensemble 4 \
    --batch_size 40 \
    --ensemble_chunk_size 4 \
    --num_steps 50 \
    --max_new_init_dates "$MAX_NEW_INITS" \
    --out_dir dataprocess/gen_multiv1_hindcast_1999_2019
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ ML hindcast generation failed with exit code $status. Not auto-resubmitting."
    exit "$status"
fi

all_done=1
for year in $(seq "$START_YEAR" "$END_YEAR"); do
    final_store="dataprocess/gen_multiv1_hindcast_1999_2019/${year}.zarr"
    if [ ! -d "$final_store" ]; then
        all_done=0
        echo "⏳ ${year} is not finished yet. Final store missing: ${final_store}"
    else
        echo "✅ ${year} final store present: ${final_store}"
    fi
done

if [ "$all_done" -eq 1 ]; then
    echo "🎉 All requested years are complete. No resubmission needed."
elif [ "$AUTO_RESUBMIT" = "1" ]; then
    echo "🔄 Preparing TACC-safe resubmission..."
    echo "📍 Current Host: $(hostname)"
    echo "📍 Submit Host: $SLURM_SUBMIT_HOST"
    echo "📍 Working Dir: $PWD"

    SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
    if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
        SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
    fi

    echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
    next_job_id=$(ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" "cd $PWD && sbatch --parsable ml_model/submit_generate_multiv1_baseline_1999_2019.sh")
    echo "🔁 Auto-resubmitted next continuation job: ${next_job_id}"
else
    echo "⏸️ AUTO_RESUBMIT=${AUTO_RESUBMIT}. Leaving follow-up submission to the user."
fi

echo "🏁 Multi-v1 ML hindcast generation job finished at $(date)"
