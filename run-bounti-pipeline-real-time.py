import os
import time
import subprocess

# Directory to monitor
base_dir = "/home/sn21/data/t2-stacks/"

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
    
command_rep = '''docker run --rm \ 
--mount type=bind,source=/home/sn21/data,target=/home/data \
 fetalsvrtk/svrtk:general_auto_amd sh -c ' \
python3 /home/auto-proc-svrtk/scripts/auto-reporting-brain-volumetry-html.py FetalMRI "{ga}" "{date}" \ 
/home/data/t2-stacks/"{path}”/grid-reo-SVR-output-brain.nii.gz  \ 
/home/data/t2-stacks/"{path}”/grid-reo-SVR-output-brain-mask-brain_bounti-19.nii.gz \
 /home/data/t2-stacks/"{path}"/brain-volumetry-report.html ; \
chmod 777 /home/data/t2-stacks/"{path}"/brain-volumetry-report.html ; ' '''

command_bio = '''docker run --rm  --mount type=bind,source=/home/sn21/data,target=/home/data \
 fetalsvrtk/svrtk:general_auto_amd sh -c ' \
mkdir /home/data/"{path}" \ 
bash /home/auto-proc-svrtk/scripts/auto-brain-biometry.sh  FetalMRI "{ga}" \
/home/data/"{path}"/grid-reo-SVR-output-brain.nii.gz  \ 
/home/data/"{path}"/grid-reo-SVR-output-brain-mask-brain_bounti-19.nii.gz \
/home/data/"{path}"/res-reo-svr-brain-output.nii.gz \ 
/home/data/"{path}"/res-reo-svr-brain-output-bounti.nii.gz \
/home/data/"{path}"/res-reo-svr-brain-output-biometry.nii.gz \ 
/home/data/"{path}"/res-reo-svr-brain-output-biometry.csv ;  \ 
python3 /home/auto-proc-svrtk/scripts/auto-reporting-brain-biometry.py FetalMRI "{ga}" "{date}"  \ 
/home/data/"{path}"/res-reo-svr-brain-output.nii.gz  \
 /home/data/"{path}"/res-reo-svr-brain-output-biometry.nii.gz \ 
/home/data/"{path}"/brain-biometry-report.html ; \ 
chmod 777  -R /home/data/"{path}”  ; ' '''

command_lung = '''docker run --rm \
    --mount type=bind,source=/home/sn21/data,target=/home/data \
    fetalsvrtk/svrtk:general_auto_amd sh -c ' \
    rm -f /home/data/t2-stacks/"{path}"/reo*; \
    chmod -R 777 /home/data/t2-stacks/"{path}"; \
    bash /home/auto-proc-svrtk/scripts/auto-lung-segmentation.sh \
    /home/data/t2-stacks/"{path}" \
    /home/data/t2-stacks/"{path}"; \
    chmod -R 777 /home/data/t2-stacks/"{path}"; \
    ' '''
 
lung_report = '''docker run --rm \ 
--mount type=bind,source=/home/sn21/data,target=/home/data \
 fetalsvrtk/svrtk:general_auto_amd sh -c ' \
python3 /home/auto-proc-svrtk/scripts/auto-reporting-brain-volumetry-html.py FetalMRI "{ga}" "{date}" \ 
/home/data/t2-stacks/"{path}"/grid-reo-DSVR-output-body.nii.gz  \ 
/home/data/t2-stacks/"{path}"/grid-reo-DSVR-output-body-mask-lung-lobes-5.nii.gz \
 /home/data/t2-stacks/"{path}"/lung-volumetry-report.html ; \
chmod 777 /home/data/t2-stacks/"{path}"/lung-volumetry-report.html ; ''''

# Record existing files at startup
existing_files = set()

def scan_existing_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "grid-reo-SVR-output-brain.nii.gz" in files:
            existing_files.add(os.path.join(root, "grid-reo-SVR-output-brain.nii.gz"))

def monitor_new_files(base_dir):
    while True:
        for root, dirs, files in os.walk(base_dir):
            file_path = os.path.join(root, "grid-reo-SVR-output-brain.nii.gz")
            if "grid-reo-SVR-output-brain.nii.gz" in files and file_path not in existing_files:
                existing_files.add(file_path)
                relative_path = os.path.relpath(root, base_dir)
                command = command_template.format(path=relative_path)
                print(f"New file detected in path: {relative_path}")
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
        time.sleep(5)  # Check every 5 seconds

# Scan existing files at initialization
scan_existing_files(base_dir)

# Start monitoring for new files
monitor_new_files(base_dir)

