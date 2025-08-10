FROM ubuntu:22.04

WORKDIR /opt/segment-anything

# first thing (because it's so big) - copy the weights file to the image
COPY sam_vit_h_4b8939.pth sam_vit_h_4b8939.pth

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install \
    torch torchvision \
    scikit-image \
    opencv-python \
    matplotlib \
    scipy \
    boto3

COPY segment_anything segment_anything
COPY find_ferrule_measurements.py find_ferrule_measurements.py
COPY find_ferrule_mask.py find_ferrule_mask.py
COPY main.py main.py
COPY masks.py masks.py
COPY local.sh local.sh
COPY entrypoint.sh entrypoint.sh

CMD ["/usr/bin/python3", "masks.py"]
