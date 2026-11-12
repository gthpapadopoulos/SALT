#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=6876159a11b4df116f30f667f8c9888617df0751
destination=${1:-third_party/dinov3}
if [[ -e "$destination" ]]; then
    current=$(git -C "$destination" rev-parse HEAD)
    [[ "$current" == "$revision" ]] || { echo "Existing checkout has a different revision" >&2; exit 1; }
else
    git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/dinov3.git "$destination"
    git -C "$destination" checkout --detach "$revision"
fi
echo "DINOv3 is ready at $destination"
