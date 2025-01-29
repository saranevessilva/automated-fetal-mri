#!/bin/bash
# This script takes a Docker image and creates a chroot image (.img)

# Syntax: ./docker_to_chroot.sh docker_image_name chroot_file_name optional_buffer_size_in_mb

if [[ $# -gt 3 ]]; then
    echo "Wrong number of arguments" >&2
    echo "Syntax: ./docker_to_chroot.sh docker_image_name chroot_file_name optional_buffer_size_in_mb" >&2
    exit 2
fi

DOCKER_NAME=${1}
CHROOT_FILE=${2}
EXPORT_FILE=docker-export.tar
BUFFER_MB=${3:-1500}  # Default to 1500MB buffer space if not provided

# Create a Docker container and export to a .tar file
echo ------------------------------------------------------------
echo Exporting Docker image ${DOCKER_NAME}
echo ------------------------------------------------------------

if test -f "${EXPORT_FILE}"; then
    echo "Warning -- ${EXPORT_FILE} exists and will be overwritten!"
    rm ${EXPORT_FILE}
fi

docker create --name tmpimage ${DOCKER_NAME}
docker export -o ${EXPORT_FILE} tmpimage
docker rm tmpimage

# Run a privileged Docker to create the chroot file
docker run -it --rm          \
           --privileged=true \
           -v $(pwd):/share  \
           ubuntu            \
           /bin/bash -c "sed -i -e 's/\r//g' /share/docker_tar_to_chroot.sh && /share/docker_tar_to_chroot.sh /share/${EXPORT_FILE} /share/${CHROOT_FILE} ${BUFFER_MB}"

# Add buffer space to the chroot image
echo ------------------------------------------------------------
echo Adding ${BUFFER_MB}MB of buffer space to the chroot image
echo ------------------------------------------------------------
dd if=/dev/zero of=${CHROOT_FILE}_buffer bs=1M count=${BUFFER_MB}
cat ${CHROOT_FILE}_buffer >> ${CHROOT_FILE}
rm ${CHROOT_FILE}_buffer

# Clean up exported tar file
rm ${EXPORT_FILE}

