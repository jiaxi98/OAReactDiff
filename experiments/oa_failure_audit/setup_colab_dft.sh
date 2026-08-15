#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f pyproject.toml ]]; then
    echo "Run this script from the OAReactDiff repository root." >&2
    exit 1
fi

MAMBA_BIN="${MAMBA_BIN:-/usr/local/bin/micromamba}"
MAMBA_PREFIX="${MAMBA_PREFIX:-/content/micromamba}"
ENV_NAME="${ENV_NAME:-oa-dft}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX="${MAMBA_PREFIX}"
"${MAMBA_BIN}" create -y -n "${ENV_NAME}" python=3.10 pip
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade "pip<26"
"${MAMBA_BIN}" run -n "${ENV_NAME}" python -m pip install \
    pyscf==2.7.0 \
    gpu4pyscf-cuda12x==1.3.0 \
    gpu4pyscf-libxc-cuda12x==0.5 \
    cupy-cuda12x==13.3.0 \
    cutensor-cu12==2.0.2

"${MAMBA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import cupy
import gpu4pyscf
import pyscf

print("pyscf", pyscf.__version__)
print("gpu4pyscf", gpu4pyscf.__version__)
print("gpu", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
PY
