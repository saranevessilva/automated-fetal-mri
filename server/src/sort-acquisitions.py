import SimpleITK as sitk
import numpy as np
import os


def load_and_sort_image(nifti_file):
    # Load the NIfTI file
    img = sitk.ReadImage(nifti_file)
    data = sitk.GetArrayFromImage(img)

    # Retrieve and modify slice thickness (assuming it's the third element in the spacing tuple)
    spacing = list(img.GetSpacing())
    original_thickness = spacing[2]
    new_thickness = 4.5
    spacing[2] = new_thickness
    img.SetSpacing(spacing)

    # Retrieve the direction (srow equivalent in SimpleITK)
    direction = np.array(img.GetDirection()).reshape((3, 3))

    # Find the largest value in the direction matrix
    largest = np.max(np.abs(direction))
    print("Largest value in direction matrix:", largest)

    # Find the index of the largest value and divide it by 2
    indices = np.where(np.abs(direction) == largest)
    direction[indices] = 4.5

    # Update the direction in the image
    img.SetDirection(direction.flatten())

    # Log the updated direction
    print("Updated direction matrix:", direction)

    # Check if slices are interleaved and combine them (example logic)
    combined = np.zeros_like(data)
    mid = (data.shape[0] + 1) // 2
    odd = data[:mid, :, :]  # Odd slices
    even = data[mid:, :, :]  # Even slices
    combined[::2, :, :] = odd
    combined[1::2, :, :] = even

    print("Combined data shape:", combined.shape)

    # Create a new NIfTI image with the updated header and sorted data
    new_img = sitk.GetImageFromArray(combined)
    new_img.SetSpacing(spacing)
    new_img.SetDirection(direction.flatten())
    new_img.SetOrigin(img.GetOrigin())

    # Save the updated NIfTI file
    new_file = nifti_file.replace('.nii.gz', '_sorted.nii.gz').replace('.nii', '_sorted.nii')
    sitk.WriteImage(new_img, new_file)
    print(f"Sorted NIfTI file saved as: {new_file}")

    return new_img, new_file


def process_folder(folder_path):
    # Loop through all files in the folder
    for nifti_file in os.listdir(folder_path):
        # Check if the file is a NIfTI file (.nii or .nii.gz)
        if (nifti_file.endswith('.nii.gz') or nifti_file.endswith('.nii')) and not nifti_file.startswith('o'):
            nifti_file_path = os.path.join(folder_path, nifti_file)
            print(f"Processing NIfTI file: {nifti_file_path}")
            load_and_sort_image(nifti_file_path)


# Set the path to the 'whole-uterus' folder
whole_uterus_folder = '/home/sn21/data/t2-stacks/test/cor-whole-uterus'

# Process all NIfTI files in the folder
process_folder(whole_uterus_folder)
