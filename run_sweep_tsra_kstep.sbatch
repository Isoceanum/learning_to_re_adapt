#!/bin/bash
# Usage:
#   sbatch run_sweep_tsra_kstep.sbatch configs/tsra_kstep_ant.yaml [max_runs] [max_total_steps] [sample_seed] [dry_run]
# Example:
#   sbatch run_sweep_tsra_kstep.sbatch configs/tsra_kstep_ant.yaml 24 50000 123 0

#SBATCH --job-name=l2ra_sweep
#SBATCH --partition=dgx2q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# --- User args ---
CONFIG=${1:?Provide config path relative to learning_to_re_adapt, e.g. configs/tsra_kstep_ant.yaml}
MAX_RUNS=${2:-24}
MAX_TOTAL_STEPS=${3:-50000}
SAMPLE_SEED=${4:-123}
DRY_RUN=${5:-0}

# --- Cluster paths ---
ROOT="/global/D1/homes/ismailou/l2ra"
REPO="${ROOT}/learning_to_re_adapt"
CONFIG_PATH="${REPO}/${CONFIG}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

# --- Activate environment ---
cd "${ROOT}"
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}:${REPO}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

# --- Derive sweep output dir ---
EXP_NAME=$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}', 'r'))['experiment_name'])")
DATE=$(date +%Y-%m-%d)
BASE_DIR="${REPO}/outputs/${DATE}"
SWEEP_DIR="${BASE_DIR}/${EXP_NAME}_sweep"
COUNTER=1
while [[ -d "${SWEEP_DIR}" ]]; do
  SWEEP_DIR="${BASE_DIR}/${EXP_NAME}_sweep_${COUNTER}"
  COUNTER=$((COUNTER+1))
done
mkdir -p "${SWEEP_DIR}"

# --- Redirect logs into sweep dir ---
exec > "${SWEEP_DIR}/out.txt" 2> "${SWEEP_DIR}/err.txt"

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Job name: ${SLURM_JOB_NAME:-n/a} | Job ID: ${SLURM_JOB_ID:-n/a}"
echo "Sweep output dir: ${SWEEP_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "max_runs=${MAX_RUNS} max_total_steps=${MAX_TOTAL_STEPS} sample_seed=${SAMPLE_SEED} dry_run=${DRY_RUN}"

nvidia-smi || true

CMD=(
  python -m learning_to_re_adapt.scripts.sweep_tsra_kstep
  "${CONFIG_PATH}"
  --output-dir "${SWEEP_DIR}"
  --max-runs "${MAX_RUNS}"
  --max-total-steps "${MAX_TOTAL_STEPS}"
  --sample-seed "${SAMPLE_SEED}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"

echo "End time: $(date)"
echo "Done. See ${SWEEP_DIR}/sweep_results.csv and ${SWEEP_DIR}/sweep_summary.yaml"

