# StreamClean pod image (code + deps only; weights come from R2 at boot).
# Built by GitHub Actions -> ghcr.io so nothing uploads from the laptop.
# Pod boot = pull this image (fast) + weights_fetch.sh from R2 (~1-2 min). No HF/pip download.
#
# Weights are NOT baked (kept ~9 GB; CI runner disk fits). weights_fetch.sh pulls them from
#   r2:suno-render/streamclean/weights/  into /workspace/DiffuEraser/weights/
# Secrets (rclone.conf) are NEVER baked; scp'd at runtime.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# --- system deps ---
RUN apt-get update -qq && apt-get install -y -qq \
      git curl wget xz-utils unzip libgl1 libglib2.0-0 python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://rclone.org/install.sh | bash > /dev/null

# --- torch 2.8 cu128 ---
RUN pip install -q torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python -c "import torch; print('TORCH', torch.__version__)"

# --- ffmpeg static, NVENC capable ---
RUN wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ff.tar.xz \
    && tar -xf /tmp/ff.tar.xz -C /tmp \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ff* /usr/local/bin/ \
    && rm -rf /tmp/ff.tar.xz /tmp/ffmpeg-master-latest-linux64-gpl \
    && ffmpeg -version | head -1

# --- python deps (system env: numpy 1.26, DiffuEraser) ---
RUN pip install -q "audio-separator[gpu]" faster-whisper speechbrain rapidocr-onnxruntime yt-dlp \
      opencv-python-headless huggingface_hub soundfile librosa imageio imageio-ffmpeg einops

# --- DiffuEraser (ProPainter code) + reqs minus torch ---
RUN git clone -q https://github.com/lixiaowen-xw/DiffuEraser.git \
    && grep -viE '^(torch|torchvision|torchaudio|xformers)([=<> ]|$)' DiffuEraser/requirements.txt > /tmp/req.txt \
    && pip install -q -r /tmp/req.txt

# --- setup2 patches: torchvision.read_video -> cv2 reader, keep priori, resize_ratio ---
RUN cd /workspace/DiffuEraser && \
    printf 'import cv2, torch, numpy as np\ndef read_video(filename, pts_unit="sec", end_pts=None, **kw):\n    c = cv2.VideoCapture(filename); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; fr = []\n    lim = int(round(end_pts * fps)) if end_pts else None\n    while True:\n        ok, f = c.read()\n        if not ok or (lim is not None and len(fr) >= lim): break\n        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))\n    return torch.from_numpy(np.stack(fr)), torch.zeros(0), {"video_fps": fps}\n' > libs_cvread.py && \
    for f in propainter/inference.py diffueraser/diffueraser.py; do \
      grep -q libs_cvread $f || sed -i "1i import libs_cvread" $f; \
      sed -i "s/torchvision\.io\.read_video(/libs_cvread.read_video(/g" $f; \
    done && \
    sed -i "s/^    os.remove(priori) # remove priori/    pass  # keep priori (streamclean)/" diffueraser/diffueraser.py && \
    (grep -q resize_ratio run_diffueraser.py || { \
      sed -i "s|    parser.add_argument(.--save_path.|    parser.add_argument(\"--resize_ratio\", type=float, default=0.6)\n&|" run_diffueraser.py; \
      sed -i "s|mask_dilation = args.mask_dilation_iter)|mask_dilation = args.mask_dilation_iter, resize_ratio=args.resize_ratio)|" run_diffueraser.py; })

# --- system onnxruntime-gpu + numpy 1.26 ---
RUN pip uninstall -y -q onnxruntime 2>/dev/null; \
    pip install -q "onnxruntime-gpu==1.22.0" "numpy==1.26.4" \
    && python -c "import torch, onnxruntime as o; print('sys ort', o.__version__, o.get_available_providers()[:2])"

# --- venv_a: audio separation (numpy>=2), isolated ---
RUN python -m venv --system-site-packages venv_a \
    && venv_a/bin/pip install -q "numpy>=2" "audio-separator[gpu]" faster-whisper speechbrain librosa soundfile "transformers>=4.45" \
    && venv_a/bin/pip uninstall -y -q onnxruntime 2>/dev/null; \
    venv_a/bin/pip install -q --ignore-installed opencv-python-headless "onnxruntime-gpu==1.22.0" \
    && venv_a/bin/python -c "import torch, onnxruntime as o, cv2, numpy; print('venv ok', torch.cuda.is_available(), o.get_available_providers()[:2], cv2.__version__, numpy.__version__)"

COPY weights_fetch.sh /workspace/weights_fetch.sh
RUN chmod +x /workspace/weights_fetch.sh && mkdir -p /workspace/jobs

# Env is fully baked; mark it so job.sh's "setup not done" guard passes. weights_fetch.sh
# (run at boot, after rclone.conf is in place) pulls the model weights from R2.
RUN echo "SETUP2_DONE (image)" > /workspace/setup2.log && echo "SETUP_DONE (image)" > /workspace/setup.log

CMD ["sleep", "infinity"]
