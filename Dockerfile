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

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/

# Install Python dependencies and check if installation succeeds
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip freeze

# Install necessary dependencies
RUN apt update && apt install -y git git-lfs && git lfs install

# Clone additional repositories
RUN mkdir -p /opt/code && \
    cd /opt/code && \
    git clone --branch python-ismrmrd-server https://github.com/saranevessilva/automated-fetal-mri.git && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir .
    
# # Set correct permissions to access the file
RUN chmod 600 /opt/code/python-ismrmrd-server/.Xauthority

# # Optionally, you can set environment variables if required
ENV XAUTHORITY=/opt/code/python-ismrmrd-server/.Xauthority

ENV DISPLAY=:0


# Set environment variables (optional, but helps avoid interactive prompts)
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server


# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y dcm2niix

# Throw an explicit error if docker build is run from the folder *containing*
# python-ismrmrd-server instead of within it (i.e. old method)
RUN if [ -d /opt/code/python-ismrmrd-server/python-ismrmrd-server ]; then echo "docker build should be run inside of python-ismrmrd-server instead of one directory up"; exit 1; fi

# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server
RUN git lfs pull

# Entry point
COPY "entrypoint.sh" /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh


CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=i2i"]

