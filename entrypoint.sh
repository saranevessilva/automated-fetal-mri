#!/bin/bash

# Create a subfolder with today's date in /home/data/t2-stacks
DATE=$(date +%Y-%m-%d)
TARGET_DIR="/home/data/t2-stacks/$DATE"

if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    echo "Created folder: $TARGET_DIR"
else
    echo "Folder already exists: $TARGET_DIR"
fi

# Execute the CMD from the Dockerfile (your Python server)
exec "$@"
