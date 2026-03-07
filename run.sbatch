#!/bin/bash
#SBATCH --job-name=l2ra
#SBATCH --partition=h200q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# dgx2q a100q hgx2q a40q h200q

# --- User settings ---
CONFIG=$1   # relative to repo root now

# --- Derive experiment name from YAML (use full path for parsing) ---
EXP_NAME=$(grep "^experiment_name" "learning_to_re_adapt/$CONFIG" | awk '{print $2}')
DATE=$(date +%Y-%m-%d)
BASE_DIR="learning_to_re_adapt/outputs/${DATE}"
EXP_DIR="${BASE_DIR}/${EXP_NAME}"
COUNTER=1

# Loop until we find a non-existing path (like _create_output_dir)
while [ -d "$EXP_DIR" ]; do
    EXP_DIR="${BASE_DIR}/${EXP_NAME}_${COUNTER}"
    COUNTER=$((COUNTER+1))
done

mkdir -p "$EXP_DIR"

# --- Redirect logs into EXP_DIR ---
exec > "${EXP_DIR}/out.txt" 2> "${EXP_DIR}/err.txt"

echo "🔧 Running on node: $(hostname)"
echo "📆 Start time: $(date)"
echo "🧪 Job name: $SLURM_JOB_NAME | Job ID: $SLURM_JOB_ID"
echo "📂 Experiment dir: $EXP_DIR"

# --- Activate environment ---
cd /global/D1/homes/ismailou/l2ra
source venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/global/D1/homes/ismailou/l2ra/learning_to_re_adapt
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
nvidia-smi


# --- Run training from repo root ---
python -m learning_to_re_adapt.scripts.run_experiment "learning_to_re_adapt/$CONFIG" --output-dir "$EXP_DIR"

NTFY_URL="https://ntfy.sh/l2ra-ebe48a314f6d2042"

{ tail -n 12 "${EXP_DIR}/out.txt" 2>/dev/null || echo "(could not read out.txt)"; } | curl -s -o /dev/null \
  -H "Title: ${EXP_NAME}" \
  -H "Content-Type: text/plain" \
  --data-binary @- \
  "$NTFY_URL"