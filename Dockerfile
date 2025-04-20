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

# Install necessary dependencies for Git LFS
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Clone the public repository (automated-fetal-mri)
RUN git clone --branch python-ismrmrd-server https://github.com/saranevessilva/automated-fetal-mri.git /opt/code/python-ismrmrd-server
WORKDIR /opt/code/python-ismrmrd-server

# Pull LFS files (since the repository is public, no authentication needed)
RUN git lfs pull

# Copy other files and dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy entry point script
COPY "entrypoint.sh" /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

# Entry point for the container
CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=i2i" ]

