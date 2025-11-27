#! /bin/bash
#SBATCH --job-name=unimat_training

set -euo pipefail

IMAGE=""
PROJECT_DIR=""
ENV_DISK=""
VENV_NAME="unimat-env"

mkdir -p "{$ENV_DISK}"
mkdir -p "{$PROJECT_DIR}"
mkdir -p "{$PROJECT_DIR}/logs"

VENV_DIR="${ENV_DISK}/${VENV_NAME}"

echo "Project: ${PROJECT_DIR}"
echo "Environment: ${VENV_DIR}"
echo "Logs: ${PROJECT_DIR}/logs"
echo "Image: ${IMAGE}"
echo "Hostname: $(hostname)"
echo "Start: $(date)"

aptainer exec \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${VENV_DIR}:/env" \
    "${IMAGE}" \
    bash -lc "

set -euo pipefail
echo 'Running node:' \$(hostname)
VENV_DIR=/envs/\${VENV_NAME}
if [ ! -d \"\$VENV_DIR\" ]; then
    echo 'Creating VENV ' \"\$VENV_DIR\"
    python -m venv \"\$VENV_DIR\"
    source \"\$VENV_DIR/bin/activate\"
    pip install --upgrade pip
    pip install torch pytorch-lightning
else
    echo 'Using existing VENV' \"\$VENV_DIR\"
    source \"\$VENV_DIR/bin/activate\"
fi

python main.py

cd /workspace

"
echo "End: $(date)"