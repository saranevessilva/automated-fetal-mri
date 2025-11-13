# =========================
# Stage 1: Build ISMRMRD and siemens_to_ismrmrd
# =========================
FROM python:3.10-bullseye AS mrd_converter

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    libboost-all-dev \
    libfftw3-dev \
    libpugixml-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/code

# Build ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout v1.13.4 && \
    mkdir build && cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install

# Build siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.10 && \
    mkdir build && cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install

# Archive ISMRMRD libraries for easier copy
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# =========================
# Stage 2: Final Python image
# =========================
FROM python:3.10-slim

LABEL org.opencontainers.image.description="Automated fetal MRI tools"
LABEL org.opencontainers.image.authors="Sara Neves Silva (sara.neves_silva@kcl.ac.uk)"

# Copy ISMRMRD libraries
COPY --from=mrd_converter /usr/local/include/ismrmrd /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd* /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz /usr/local/lib/

RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd

# Install runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxslt1.1 \
    libhdf5-dev \
    libboost-program-options-dev \
    libpugixml-dev \
    dos2unix \
    nano \
    git \
    git-lfs \
    dcm2niix \
    && git lfs install \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip freeze

# Clone additional repositories
RUN mkdir -p /opt/code && cd /opt/code && \
    git clone --branch landmarks https://github.com/saranevessilva/automated-fetal-mri.git && \
    git clone https://github.com/kspacekelvin/python-ismrmrd-server.git && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd ismrmrd-python-tools && pip3 install --no-cache-dir . && pip freeze

# Set permissions for Xauthority
RUN chmod 600 /opt/code/automated-fetal-mri/.Xauthority

# Environment variables
ENV XAUTHORITY=/opt/code/automated-fetal-mri/.Xauthority
ENV DISPLAY=:0
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /opt/code/automated-fetal-mri

# Pull Git LFS files
RUN git lfs pull

# Entry point
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/usr/local/bin/entrypoint.sh"]
CMD ["python3", "main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]

