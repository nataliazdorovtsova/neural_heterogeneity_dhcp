#!/bin/bash
# Build the project environment at env/ (self-contained, on /imaging/astle).
#
#   bash scripts/setup_env.sh
#
# Deliberately a conda prefix rather than ~/.conda/envs: it keeps the env next
# to the code, off the 50 GB home quota, and makes the whole project one
# directory to copy or delete.
#
# conda is used only to bootstrap python + pip (the site conda is 4.10.1 and
# the site mamba 0.13.0 — both ancient); every real package comes from pip,
# pinned in requirements.txt.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$PROJECT/env"
CONDA=/imaging/local/software/miniconda/bin/conda

# ~/.condarc sets ssl_verify to /etc/ssl/certs/ca-bundle.crt, which is a RHEL
# path that does not exist on these Ubuntu nodes. Override it here rather than
# editing the global config, which other projects may depend on.
export CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt

if [[ -d "$ENV_DIR" ]]; then
    echo "env/ already exists — remove it first to rebuild:"
    echo "    rm -rf $ENV_DIR"
    exit 1
fi

echo "==> creating python 3.12 env at $ENV_DIR"
"$CONDA" create -y -q -p "$ENV_DIR" python=3.12 pip

echo "==> upgrading pip"
"$ENV_DIR/bin/python" -m pip install --no-input -q --upgrade pip

echo "==> installing pinned packages"
"$ENV_DIR/bin/python" -m pip install --no-input -r "$PROJECT/requirements.txt"

echo "==> registering jupyter kernel"
"$ENV_DIR/bin/python" -m ipykernel install --user \
    --name neural-heterogeneity-dhcp \
    --display-name "neural heterogeneity (dHCP)"

echo
echo "==> done. activate with:"
echo "    source $PROJECT/activate.sh      # bash"
echo "    source $PROJECT/activate.csh     # tcsh"
