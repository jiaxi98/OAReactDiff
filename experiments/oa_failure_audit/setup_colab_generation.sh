#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml || ! -f pretrained-ts1x-diff.ckpt ]]; then
    echo "Run this script from the OAReactDiff repository root." >&2
    exit 1
fi

MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_PREFIX="${MAMBA_PREFIX:-/content/micromamba}"
ENV_NAME="${ENV_NAME:-oa-generation}"
ENV_PREFIX="${MAMBA_PREFIX}/envs/${ENV_NAME}"

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

# Colab's system libstdc++ is older than the conda-forge ICU build. Ensure
# subprocesses resolve the environment runtime before the system copy.
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install --upgrade \
    "pip<26" "setuptools<70" wheel
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    torch-scatter==2.1.0 torch-sparse==0.6.16 \
    -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    torch-geometric==2.2.0 \
    numpy==1.24.4 \
    pandas==1.5.3 \
    pytorch-lightning==1.8.6 \
    torchmetrics==0.11.4 \
    pymatgen==2023.5.10
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install -e . --no-deps

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python - <<'PY'
import sqlite3
import sys
from pathlib import Path

import torch
import torch_geometric
import torch_scatter
from oa_reactdiff.trainer.pl_trainer import DDPMModule

assert torch.cuda.is_available(), "Colab GPU is not visible to PyTorch"
libstdcxx_paths = sorted({
    line.rsplit(maxsplit=1)[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "libstdc++.so.6" in line
})
assert libstdcxx_paths and all(
    Path(path).is_relative_to(sys.prefix) for path in libstdcxx_paths
), f"Wrong libstdc++ loaded: {libstdcxx_paths}"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torch_geometric", torch_geometric.__version__)
print("torch_scatter", torch_scatter.__version__)
print("sqlite", sqlite3.sqlite_version)
print("libstdc++", ", ".join(libstdcxx_paths))
print("gpu", torch.cuda.get_device_name(0))
print("DDPMModule import: ok")
PY
