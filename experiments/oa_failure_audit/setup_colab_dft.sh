#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
    echo "Run this script from the OAReactDiff repository root." >&2
    exit 1
fi

MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_PREFIX="${MAMBA_PREFIX:-/content/micromamba}"
ENV_NAME="${ENV_NAME:-oa-dft}"
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

# Prefer the environment's GNU runtime over Colab's older system copy.
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install --upgrade "pip<26"
"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python -m pip install \
    pyscf==2.7.0 \
    gpu4pyscf-cuda12x==1.3.0 \
    gpu4pyscf-libxc-cuda12x==0.5 \
    cupy-cuda12x==13.3.0 \
    cutensor-cu12==2.0.2

"${MAMBA_BIN}" run -p "${ENV_PREFIX}" python - <<'PY'
import sqlite3
import sys
from pathlib import Path

import cupy
import gpu4pyscf
import pyscf

libstdcxx_paths = sorted({
    line.rsplit(maxsplit=1)[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "libstdc++.so.6" in line
})
assert libstdcxx_paths and all(
    Path(path).is_relative_to(sys.prefix) for path in libstdcxx_paths
), f"Wrong libstdc++ loaded: {libstdcxx_paths}"
print("pyscf", pyscf.__version__)
print("gpu4pyscf", gpu4pyscf.__version__)
print("sqlite", sqlite3.sqlite_version)
print("libstdc++", ", ".join(libstdcxx_paths))
print("gpu", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
PY
