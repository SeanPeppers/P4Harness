#!/bin/bash
# Example Slurm job for the operational suite (Exp 1-3) on a GPU node with Apptainer.
# Edit the #SBATCH lines and paths for your cluster before submitting.
#SBATCH --job-name=p4harness
#SBATCH --output=p4harness_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00
set -euo pipefail

module load apptainer 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OLLAMA_MODELS=${OLLAMA_MODELS:-$PWD/.ollama}
export P4C_SIF=${P4C_SIF:-$PWD/p4c.sif}      # apptainer build p4c.sif containers/p4c.def
OLLAMA_SIF=${OLLAMA_SIF:-$PWD/ollama.sif}    # apptainer build ollama.sif containers/ollama.def
mkdir -p "$OLLAMA_MODELS"

ollama() { apptainer exec --nv --bind "$OLLAMA_MODELS:/root/.ollama" "$OLLAMA_SIF" ollama "$@"; }
setsid apptainer exec --nv --bind "$OLLAMA_MODELS:/root/.ollama" "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!
trap 'kill -- "-$OLLAMA_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 24); do
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2)" 2>/dev/null && break
    sleep 5
done
ollama pull qwen2.5-coder:14b

python experiments/runner.py run exp1
python experiments/runner.py run exp2
python experiments/runner.py run exp3
python experiments/runner.py report
