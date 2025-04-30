FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System-Updates and basics
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Python-Dependencies # further export in req.txt
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install gradio diffusers transformers accelerate xformers

# Workspace
WORKDIR /workspace
COPY App/app.py .

# run
CMD ["python", "app.py"]