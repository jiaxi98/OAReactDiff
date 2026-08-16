#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
    echo "Run this script from the OAReactDiff repository root." >&2
    exit 1
fi

MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_PREFIX="${MAMBA_PREFIX:-/content/micromamba}"
ENV_NAME="${ENV_NAME:-oa-horm}"
ENV_PREFIX="${MAMBA_PREFIX}/envs/${ENV_NAME}"
HORM_DIR="${HORM_DIR:-/content/HORM}"
HORM_COMMIT="b4c2a35a28985c72ca47261bad0a96b2bc2ba084"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${MAMBA_PREFIX}"
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    "${MAMBA_BIN}" install -y -p "${ENV_PREFIX}" --channel conda-forge \
        "libstdcxx-ng>=15" "libgcc-ng>=15"
else
    "${MAMBA_BIN}" create -y -p "${ENV_PREFIX}" --channel conda-forge \
        python=3.10 pip "libstdcxx-ng>=15" "libgcc-ng>=15"
fi

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install --upgrade \
    "pip<26" "setuptools<70" wheel
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    torch-geometric==2.6.1 \
    numpy==1.26.4 \
    scipy==1.13.1 \
    matplotlib==3.9.0 \
    networkx==3.4.2 \
    pytorch-lightning==2.4.0 \
    torchmetrics==1.4.0 \
    e3nn==0.4.4 \
    opt-einsum-fx==0.1.4 \
    lmdb==1.5.1 \
    pyyaml

if [[ ! -d "${HORM_DIR}/.git" ]]; then
    git clone https://github.com/deepprinciple/HORM.git "${HORM_DIR}"
fi
git -C "${HORM_DIR}" fetch --depth 1 origin "${HORM_COMMIT}"
git -C "${HORM_DIR}" checkout --detach "${HORM_COMMIT}"
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install -e "${HORM_DIR}" --no-deps

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python - <<'PY'
import sqlite3

import numpy
import scipy
import torch
import torch_geometric
import torch_scatter

assert torch.cuda.is_available(), "Colab GPU is not visible to PyTorch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("torch_geometric", torch_geometric.__version__)
print("torch_scatter", torch_scatter.__version__)
print("sqlite", sqlite3.sqlite_version)
print("gpu", torch.cuda.get_device_name(0))
PY
