# Stage 1: Build ISMRMRD and siemens_to_ismrmrd
FROM python:3.10.2-slim AS mrd_converter
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/code

# Build ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout v1.13.4 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install

# Build siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.10 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install

# Create ISMRMRD archive
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# Use Docker-in-Docker image
FROM docker:latest

# Install dependencies
RUN apk add --no-cache \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    sudo

# Enable Docker daemon
RUN dockerd &

# Pull the image
RUN apt-get update && apt-get install -y docker.io
RUN docker pull fetalsvrtk/svrtk:general_auto_amd

# Stage 2: Final Image
FROM python:3.10.2-slim
LABEL org.opencontainers.image.description="Automated fetal MRI tools"
LABEL org.opencontainers.image.authors="Sara Neves Silva (sara.neves_silva@kcl.ac.uk)"

# Copy ISMRMRD libraries
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxslt1.1 \
    libhdf5-dev \
    libboost-program-options-dev \
    libpugixml-dev \
    dos2unix \
    nano \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/

# Install Python dependencies and check if installation succeeds
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip freeze

# Install PyTorch with CUDA support (1.10.0 with CUDA 11.3)
# RUN pip install torch==2.5.1 && pip install torchvision==0.15.1

# # Install a specific version of nnUNet
# RUN git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/code/nnUNet && \
#     cd /opt/code/nnUNet && \
#     pip install -e .

# Install necessary dependencies
RUN apt update && apt install -y git git-lfs && git lfs install

# Clone additional repositories
RUN mkdir -p /opt/code && \
    cd /opt/code && \
    git clone https://github.com/kspacekelvin/python-ismrmrd-server.git && \
    git clone https://github.com/saranevessilva/automated-fetal-mri.git && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir . && \
    pip freeze

# Set correct permissions to access the file
RUN chmod 600 /opt/code/automated-fetal-mri/.Xauthority

# Optionally, you can set environment variables if required
ENV XAUTHORITY=/opt/code/automated-fetal-mri/.Xauthority

ENV DISPLAY=:0

# Set environment variables (optional, but helps avoid interactive prompts)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    dcmtk \
    curl \
    wget \
    unzip \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*
    
# Download, extract, and install the latest dcm2niix
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip && \
    unzip dcm2niix_lnx.zip -d dcm2niix && \
    mv dcm2niix/dcm2niix /usr/local/bin/ && \
    rm -rf dcm2niix_lnx.zip dcm2niix
    
# Install dependencies
RUN apt-get update && apt-get install -y \
    wget unzip tar && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/code/automated-fetal-mri
RUN git lfs pull

# Entry point
COPY "entrypoint.sh" /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]

CMD ["python3", "main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]

