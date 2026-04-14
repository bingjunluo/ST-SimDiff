#!/bin/bash
# Run after pip install -e . completes
# Installs additional required packages

set -e

PYTHON=/root/miniconda3/envs/exp1/bin/python
PIP=/root/miniconda3/envs/exp1/bin/pip
MIRROR=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/

echo "=== Installing transformers==4.51.3 (as specified in README) ==="
${PIP} install transformers==4.51.3 -i ${MIRROR} 2>&1

echo "=== Installing networkx (required by simdiff) ==="
${PIP} install networkx -i ${MIRROR} 2>&1

echo "=== Installing additional packages ==="
${PIP} install flash-attn --no-build-isolation -i ${MIRROR} 2>&1 || echo "flash-attn not available, using sdpa instead (OK)"

echo "=== Verifying key packages ==="
${PYTHON} -c "
import torch; print('torch:', torch.__version__)
import transformers; print('transformers:', transformers.__version__)
import accelerate; print('accelerate:', accelerate.__version__)
import decord; print('decord: OK')
import cv2; print('cv2:', cv2.__version__)
import networkx; print('networkx:', networkx.__version__)
import einops; print('einops: OK')
import lmms_eval; print('lmms_eval: OK')
"

echo "=== Verifying llava and simdiff ==="
cd /root/ST-SimDiff-main
${PYTHON} -c "
import sys
sys.path.insert(0, '/root/ST-SimDiff-main')
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
print('llava: OK')
from simdiff.interface import apply_simdiff
print('simdiff: OK')
from simdiff.new_topk import SimDiff
print('SimDiff (new_topk): OK')
"

echo "=== Setup complete ==="
