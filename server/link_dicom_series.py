import os
import pydicom
import warnings

def link_dicoms_into_series(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store DICOM files organized by series
    series_dict = {}

    # Iterate through all files in the input folder
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith('.dcm'):
                dicom_filepath = os.path.join(root, filename)

                # Suppress specific warnings related to the "UI" value representation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dicom_data = pydicom.dcmread(dicom_filepath, force=True)

                # Extract series instance UID
                series_uid = str(dicom_data.SeriesInstanceUID)

                # Organize DICOM files by series
                if series_uid not in series_dict:
                    series_dict[series_uid] = []
                series_dict[series_uid].append(dicom_filepath)

    # Iterate through the organized series and copy/link DICOM files
    for series_uid, dicom_files in series_dict.items():
        series_folder = os.path.join(output_folder, series_uid)

        # Create a folder for each series
        os.makedirs(series_folder, exist_ok=True)

        # Link or copy DICOM files to the series folder
        for dicom_filepath in dicom_files:
            output_filepath = os.path.join(series_folder, os.path.basename(dicom_filepath))
            os.symlink(dicom_filepath, output_filepath)  # Use os.link for Windows

if __name__ == "__main__":
    input_folder = "/home/sn21/Desktop/haste-raw/dicoms"
    output_folder = "/home/sn21/Desktop/haste-raw/dicoms/series"

    link_dicoms_into_series(input_folder, output_folder)
