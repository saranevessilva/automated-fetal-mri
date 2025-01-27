import os
import subprocess
import glob


def convert_dicom_to_nifti(folder_path, output_folder):
    # Extract the folder name
    folder_name = os.path.basename(folder_path)

    # Run dcm2nii command and specify the output directory for NIfTI files
    subprocess.run(['dcm2nii', '-g', 'n', '-o', output_folder, folder_path])

    # Find the generated NIfTI file in the output folder (not within subfolder)
    nifti_files = glob.glob(os.path.join(output_folder, f'{folder_name}*.nii'))

    if nifti_files:
        nifti_file = nifti_files[0]

        # Construct the new NIfTI file path (using folder_name as the name)
        new_nifti_path = os.path.join(output_folder, f"{folder_name}.nii")

        # Rename the NIfTI file to match the folder name
        os.rename(nifti_file, new_nifti_path)
        print(f"NIfTI file renamed to {new_nifti_path}")

    else:
        print(f"No NIfTI file found in {folder_path}")


# Set the base test folder path
base_test_folder = '/home/sn21/data/t2-stacks/test/dicoms'
output_folder = '/home/sn21/data/t2-stacks/test/cor-whole-uterus'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all subdirectories in the test folder
for folder_name in os.listdir(base_test_folder):
    folder_path = os.path.join(base_test_folder, folder_name)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")
        convert_dicom_to_nifti(folder_path, output_folder)
