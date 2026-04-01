#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs"
PYTHON_BIN="${PYTHON_BIN:-python3}"

configs=(
  "$CONFIG_DIR/meta_lora_4_lr_0001.yaml"
  "$CONFIG_DIR/meta_lora_4_lr_0003.yaml"
  "$CONFIG_DIR/meta_lora_4_lr_001.yaml"
  "$CONFIG_DIR/meta_lora_8_lr_0001.yaml"
  "$CONFIG_DIR/meta_lora_8_lr_0003.yaml"
  "$CONFIG_DIR/meta_lora_8_lr_001.yaml"
  "$CONFIG_DIR/meta_lora_16_lr_0001.yaml"
  "$CONFIG_DIR/meta_lora_16_lr_0003.yaml"
  "$CONFIG_DIR/meta_lora_16_lr_001.yaml"
)

for cfg in "${configs[@]}"; do
  echo "=== Running $(basename "$cfg") ==="
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_experiment.py" "$cfg"
done
