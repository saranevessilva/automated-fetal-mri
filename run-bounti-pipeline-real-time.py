import os
import subprocess
import glob

# Directory to search
base_dir = "/tmp/share/debug/"

# Docker command template
command_template = '''docker run --rm \
    --mount type=bind,source=/home/sn21/data,target=/home/data \
    fetalsvrtk/svrtk:general_auto_amd sh -c ' \
    rm -f /home/data/t2-stacks/"{path}"/reo*; \
    chmod -R 777 /home/data/t2-stacks/"{path}"; \
    bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh \
    /home/data/t2-stacks/"{path}" \
    /home/data/t2-stacks/"{path}"; \
    chmod -R 777 /home/data/t2-stacks/"{path}"; \
    ' '''

def find_latest_file(base_dir):
    """Find the latest grid-reo-SVR-output-brain.nii.gz file in the directory."""
    files = glob.glob(os.path.join(base_dir, "**/grid-reo-SVR-output-brain.nii.gz"), recursive=True)
    
    if not files:
        print("No 'grid-reo-SVR-output-brain.nii.gz' files found.")
        return None
    
    latest_file = max(files, key=os.path.getmtime)  # Get the most recently modified file
    return latest_file

# Find the latest file
latest_file = find_latest_file(base_dir)

if latest_file:
    relative_path = os.path.relpath(os.path.dirname(latest_file), base_dir)  # Get folder path
    command = command_template.format(path=relative_path)
    print(f"Processing latest file: {latest_file}")
    subprocess.Popen(command, shell=True)
else:
    print("No valid files found for processing.")

