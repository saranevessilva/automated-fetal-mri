#!/bin/bash

# Create a subfolder with today's date in /home/data/t2-stacks
DATE=$(date +%Y-%m-%d)
TARGET_DIR="/home/data/t2-stacks/$DATE"
TARGET_DIR_EAGLE="/home/data/eagle/$DATE"
TARGET_DIR_OWL="/home/data/owl/$DATE"
TARGET_DIR_VOL="/home/data/eagle/$DATE"
TARGET_DIR_RAT="/home/data/rat/$DATE"

chmod 600 /opt/code/automated-fetal-mri/.Xauthority
export XAUTHORITY=/opt/code/automated-fetal-mri/.Xauthority


if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    echo "Created folder: $TARGET_DIR"
else
    echo "Folder already exists: $TARGET_DIR"
fi

if [ ! -d "$TARGET_DIR_EAGLE" ]; then
    mkdir -p "$TARGET_DIR_EAGLE"
    echo "Created folder: $TARGET_DIR_EAGLE"
else
    echo "Folder already exists: $TARGET_DIR_EAGLE"
fi

if [ ! -d "$TARGET_DIR_OWL" ]; then
    mkdir -p "$TARGET_DIR_OWL"
    echo "Created folder: $TARGET_DIR_OWL"
else
    echo "Folder already exists: $TARGET_DIR_OWL"
fi

if [ ! -d "$TARGET_DIR_VOL" ]; then
    mkdir -p "$TARGET_DIR_VOL"
    echo "Created folder: $TARGET_DIR_VOL"
else
    echo "Folder already exists: $TARGET_DIR_VOL"
fi

if [ ! -d "$TARGET_DIR_RAT" ]; then
    mkdir -p "$TARGET_DIR_RAT"
    echo "Created folder: $TARGET_DIR_RAT"
else
    echo "Folder already exists: $TARGET_DIR_RAT"
fi

# Execute the CMD from the Dockerfile (your Python server)
exec "$@"
