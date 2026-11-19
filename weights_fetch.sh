#!/bin/bash
# Pull model weights from the R2 cache into DiffuEraser/weights.
# Run at pod boot AFTER /root/.config/rclone/rclone.conf is in place (scp'd by the laptop).
set -e
mkdir -p /workspace/DiffuEraser/weights
rclone copy r2:suno-render/streamclean/weights /workspace/DiffuEraser/weights --transfers 8 --s3-chunk-size 64M
echo "WEIGHTS_READY ($(du -sh /workspace/DiffuEraser/weights | cut -f1))"
