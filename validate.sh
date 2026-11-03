#!/usr/bin/env bash
# Usage : ./validate.sh [bits]   (par défaut 8)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BITS="${1:-8}"
IMAGE="nn2fpga-container-image"
CONFIG="/workspace/config/mnist_zcu102_check.toml"
"$ROOT/.venv/bin/python" "$ROOT/src/export_onnx.py" --bits "$BITS"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Construction de l'image $IMAGE..."
    docker build --file "$ROOT/NN2FPGA/docker/Dockerfile" \
        --build-arg NN2FPGA_ROOT_DIR=/workspace/NN2FPGA \
        -t "$IMAGE" "$ROOT/NN2FPGA"
fi

docker run --rm \
    --entrypoint "" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e XILINX_VERSION=2024.2 \
    -e NN2FPGA_ROOT_DIR=/workspace/NN2FPGA \
    -v "$ROOT:/workspace" \
    -w /workspace/NN2FPGA \
    "$IMAGE" \
    python3 -m nn2fpga.compiler.cli --config "$CONFIG"
