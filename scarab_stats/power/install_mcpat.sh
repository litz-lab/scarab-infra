#!/usr/bin/env bash
#
# Build McPAT from source and install the binary at scarab_stats/power/mcpat.
#
# The pipeline pins HewlettPackard/mcpat at v1.3.0
# (commit 74d4759f3ba2dff8f5a69e07a68efdb46b42fb8c, Feb 2015 banner),
# matching the McPAT release the converter and field-mapping tables
# under template/ were authored against.
#
# Usage:
#     ./install_mcpat.sh                       # install to default path
#     MCPAT_BIN=/path/to/mcpat ./install_mcpat.sh   # also symlink override path
#
set -euo pipefail

REPO_URL="https://github.com/HewlettPackard/mcpat.git"
PINNED_REF="74d4759f3ba2dff8f5a69e07a68efdb46b42fb8c"  # v1.3.0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/mcpat_src"
TARGET="${MCPAT_BIN:-${SCRIPT_DIR}/mcpat}"

if [[ -x "${TARGET}" ]]; then
    echo "[install_mcpat] ${TARGET} already exists; skipping (rm to force rebuild)"
    exit 0
fi

echo "[install_mcpat] cloning ${REPO_URL} into ${BUILD_DIR}"
if [[ ! -d "${BUILD_DIR}/.git" ]]; then
    git clone "${REPO_URL}" "${BUILD_DIR}"
fi
git -C "${BUILD_DIR}" fetch --all --quiet
git -C "${BUILD_DIR}" checkout --quiet "${PINNED_REF}"

echo "[install_mcpat] building (this may take a minute)"
make -C "${BUILD_DIR}" -j"$(nproc)"

if [[ ! -x "${BUILD_DIR}/mcpat" ]]; then
    echo "[install_mcpat] build did not produce mcpat binary" >&2
    exit 1
fi

mkdir -p "$(dirname "${TARGET}")"
cp -f "${BUILD_DIR}/mcpat" "${TARGET}"
echo "[install_mcpat] installed at ${TARGET}"
"${TARGET}" 2>&1 | head -1 || true
