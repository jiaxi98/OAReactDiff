#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml || ! -f pretrained-ts1x-diff.ckpt ]]; then
    echo "Run this script from the OAReactDiff repository root." >&2
    exit 1
fi

MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_PREFIX="${MAMBA_PREFIX:-/content/micromamba}"
ENV_NAME="${ENV_NAME:-oa-generation}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${MAMBA_PREFIX}"
"${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.10 pip
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade \
    "pip<26" "setuptools<70" wheel
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    torch-scatter==2.1.0 torch-sparse==0.6.16 \
    -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    torch-geometric==2.2.0 \
    numpy==1.24.4 \
    pandas==1.5.3 \
    pytorch-lightning==1.8.6 \
    torchmetrics==0.11.4 \
    pymatgen==2023.5.10
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install -e . --no-deps

"${MAMBA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import torch
import torch_geometric
import torch_scatter
from oa_reactdiff.trainer.pl_trainer import DDPMModule

assert torch.cuda.is_available(), "Colab GPU is not visible to PyTorch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_geometric", torch_geometric.__version__)
print("torch_scatter", torch_scatter.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("DDPMModule import: ok")
PY
