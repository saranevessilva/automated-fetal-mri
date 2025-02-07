import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter

# import numpy as np
import ismrmrd
import logging
import time
import io
import os
from datetime import datetime
import subprocess
import matplotlib
#
from scipy.ndimage import map_coordinates

from ismrmrd.meta import Meta
import itertools
import ctypes
# import numpy as np
import copy
import glob
import warnings
from scipy import ndimage, misc
from skimage import measure
from scipy.spatial.distance import euclidean

warnings.simplefilter('default')

from ismrmrd.acquisition import Acquisition
from ismrmrd.flags import FlagsMixin
from ismrmrd.equality import EqualityMixin
from ismrmrd.constants import *

import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import sys

import nibabel as nib
import SimpleITK as sitk
from numpy.fft import fftshift, ifftshift, fftn, ifftn

# standard libraries
from matplotlib import pyplot as plt

import os

# import numpy as np
# import matplotlib as plt
import math
import scipy
from skimage import io

from scipy import ndimage
from skimage.measure import label, regionprops
import argparse
from src.mrd2dicom_svr import main

import warnings

warnings.filterwarnings("ignore")


# Reset and configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from scipy.ndimage import affine_transform
except ImportError:
    print("Error: scipy is not installed or affine_transform is not available")

# Folder for debug output files
debugFolder = "/tmp/share/debug"
date_path = datetime.today().strftime("%Y-%m-%d")
os.environ['PATH'] += ':/usr/local/bin'


def load_and_sort_image(nifti_file, new_thickness):
    # Load the NIfTI file
    img = sitk.ReadImage(nifti_file)
    data = sitk.GetArrayFromImage(img)

    # Retrieve and modify slice thickness (assuming it's the third element in the spacing tuple)
    spacing = list(img.GetSpacing())
    original_thickness = spacing[2]
    spacing[2] = new_thickness
    img.SetSpacing(spacing)

    # Retrieve the direction (srow equivalent in SimpleITK)
    direction = np.array(img.GetDirection()).reshape((3, 3))

    # Find the largest value in the direction matrix
    largest = np.max(np.abs(direction))
    print("Largest value in direction matrix:", largest)

    # Find the index of the largest value and divide it by 2
    indices = np.where(np.abs(direction) == largest)
    direction[indices] = new_thickness

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
    # new_file = nifti_file.replace('.nii.gz', '_sorted.nii.gz').replace('.nii', '_sorted.nii')
    sitk.WriteImage(new_img, nifti_file)
    print(f"Sorted NIfTI file saved as: {nifti_file}")

    return new_img


def process_folder(folder_path, pixdim_z):
    # Loop through all files in the folder
    for nifti_file in os.listdir(folder_path):
        # Check if the file is a NIfTI file (.nii or .nii.gz)
        if (nifti_file.endswith('.nii.gz') or nifti_file.endswith('.nii')) and not nifti_file.startswith('o'):
            nifti_file_path = os.path.join(folder_path, nifti_file)
            print(f"Processing NIfTI file: {nifti_file_path}")
            load_and_sort_image(nifti_file_path, pixdim_z)


def process_mrd_files(folder_path, in_group=None, out_folder=None):
    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.startswith('haste_') and file_name.endswith('.h5'):
            file_path = os.path.join(folder_path, file_name)
            print("MRD file:", file_path)

            # Prepare arguments for the main function
            args = argparse.Namespace(
                filename=file_path,
                in_group=in_group,
                out_folder=out_folder or os.path.join(folder_path, os.path.splitext(file_name)[0])
            )

            # Call the main function
            main(args)


def convert_to_nii(folder_path, output_folder):
    # Extract the folder name
    folder_name = os.path.basename(folder_path)

    # subprocess.run(['dcm2niix', '-g', 'n', '-o', output_folder, folder_path])
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


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3, "
                     "a matrix size of (%s x %s x %s), %s slices and %s echoes",
                     metadata.encoding[0].trajectory,
                     metadata.encoding[0].encodedSpace.matrixSize.x,
                     metadata.encoding[0].encodedSpace.matrixSize.y,
                     metadata.encoding[0].encodedSpace.matrixSize.z,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
                     metadata.encoding[0].encodedSpace.fieldOfView_mm.z,
                     metadata.encoding[0].encodingLimits.slice.maximum + 1,
                     metadata.encoding[0].encodingLimits.contrast.maximum + 1)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:

            state = {
                "slice_pos": 0,
                "min_slice_pos": 0,
                "first_slice": 1
            }

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                        not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                        not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
                        not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)

                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, connection, config, metadata)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d",
                                 item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata, state)
                    connection.send_image(image)
                    imgGroup = []

                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry'] = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key=lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData, 1)

        # Process any remaining groups of raw or image data.  This can
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, connection, config, metadata)
            connection.send_image(image)
            acqGroup = []

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata, state)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()


def process_raw(group, connection, config, metadata):
    if len(group) == 0:
        return []

    # Start timer
    tic = perf_counter()

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO phs] array
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    phs = [acquisition.idx.phase for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0],
                     metadata.encoding[0].encodedSpace.matrixSize.y,
                     metadata.encoding[0].encodedSpace.matrixSize.x,
                     max(phs) + 1),
                    group[0].data.dtype)

    rawHead = [None] * (max(phs) + 1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:, lin, -acq.data.shape[1]:, phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (
                    np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(
                    rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Remove readout oversampling
    data = fft.ifft(data, axis=2)
    data = np.delete(data, np.arange(int(data.shape[2] * 1 / 4), int(data.shape[2] * 3 / 4)), 2)
    data = fft.fft(data, axis=2)

    logging.debug("Raw data is size after readout oversampling removal %s" % (data.shape,))
    np.save(debugFolder + "/" + "rawNoOS.npy", data)

    # Fourier Transform
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    # Data will be [PE RO phs]
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32767 / data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x) / 2)
    data = data[:, offset:offset + metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y) / 2)
    data = data[offset:offset + metadata.encoding[0].reconSpace.matrixSize.y, :]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Measure processing time
    toc = perf_counter()
    strProcessTime = "Total processing time: %.2f ms" % ((toc - tic) * 1000.0)
    logging.info(strProcessTime)

    # Send this as a text message back to the client
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # Format as ISMRMRD image data
    imagesOut = []
    for phs in range(data.shape[2]):
        # Create new MRD instance for the processed image
        # data has shape [PE RO phs], i.e. [y x].
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        tmpImg = ismrmrd.Image.from_array(data[..., phs], transpose=False)

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole'] = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter'] = '16384'
        tmpMeta['WindowWidth'] = '32768'
        tmpMeta['Keep_image_geometry'] = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to invert image contrast
    imagesOut = process_image(imagesOut, connection, config, metadata)

    return imagesOut


def process_image(images, connection, config, metadata, state):
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images),
                  ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data for img in images])
    head = [img.getHead() for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    imheader = head[0]

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    ncontrasts = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    nreps = metadata.encoding[0].encodingLimits.repetition.maximum + 1
    ninstances = nslices * ncontrasts * nreps

    print("Number of echoes =", ncontrasts)
    print("Number of instances =", ninstances)

    pixdim_x = (metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x)
    pixdim_y = metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[0].encodedSpace.matrixSize.y
    pixdim_z = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    print("pixdims", pixdim_x, pixdim_y, pixdim_z)

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    print("Reformatted data", data.shape)

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= 32767 / data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Invert image contrast
    # data = 32767-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    im = np.squeeze(data)
    im = nib.Nifti1Image(im, np.eye(4))
    nib.save(im, debugFolder + "/" + "im.nii.gz")

    slice = imheader.slice
    contrast = imheader.contrast
    repetition = imheader.repetition
    print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

    svr_path = (debugFolder + "/" + date_path)

    # Check if the parent directory exists, if not, create it
    if not os.path.exists(svr_path):
        os.makedirs(svr_path)

    process_mrd_files(svr_path)  # Conversion of MRD to DICOM happens here

    # Set the base test folder path
    output_folder = svr_path + "/dicoms"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all subdirectories in the test folder
    for folder_name in os.listdir(svr_path):
        folder_path = os.path.join(svr_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            convert_to_nii(folder_path, output_folder)

    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)

        # Debugging step: print the file names
        print(f"Checking file: {file_name}")

        if os.path.isfile(file_path) and file_name.startswith('o'):
            try:
                print(f"Deleting file: {file_path}")
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_name}: {e}")

    # Process all NIfTI files in the folder
    process_folder(output_folder, pixdim_z)

    print("Launching docker now...")

    # Set the DISPLAY and XAUTHORITY environment variables
    os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
    # os.environ['XAUTHORITY'] = '/home/sn21/.Xauthority'
    os.environ['XAUTHORITY'] = "/opt/code/automated-fetal-mri/.Xauthority"

    command = f'''docker run --rm --mount type=bind,source=/tmp/share/debug,target=/home/data \
    fetalsvrtk/svrtk:general_auto_amd sh -c 'bash /home/auto-proc-svrtk/scripts/auto-brain-055t-reconstruction.sh \
    /home/data/{date_path}/dicoms /home/data/{date_path}/{date_path}-result 1 4.5 1.0 1 ; \
    chmod 1777 -R /home/data/{date_path}/{date_path}-result ; \
    /bin/MIRTK/build/lib/tools/pad-3d /home/data/{date_path}/{date_path}-result/reo-SVR-output-brain.nii.gz /home/ref.nii.gz 160 1 ; \
    /bin/MIRTK/build/lib/tools/edit-image /home/ref.nii.gz /home/ref.nii.gz -dx 1 -dy 1 -dz 1 ; \
    /bin/MIRTK/build/lib/tools/transform-image /home/data/{date_path}/{date_path}-result/reo-SVR-output-brain.nii.gz \
    /home/data/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -target /home/ref.nii.gz -interp BSpline ; \
    /bin/MIRTK/build/lib/tools/nan /home/data/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz 1000000 ; \
    /bin/MIRTK/build/lib/tools/convert-image /home/data/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz \
    /home/data/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -short ; \
    chmod 1777 /home/data/{date_path}/{date_path}-result/grid-reo-SVR-output-brain.nii.gz ; \
    bash /home/auto-proc-svrtk/scripts/auto-body-055t-reconstruction.sh /home/data/{date_path}/dicoms \
    /home/data/{date_path}/{date_path}-result 1 4.5 1.0 1 ; \
    chmod 1777 -R /home/data/{date_path}/{date_path}-result ; \
    /bin/MIRTK/build/lib/tools/pad-3d /home/data/{date_path}/{date_path}-result/reo-DSVR-output-body.nii.gz /home/ref.nii.gz 256 1 ; \
    /bin/MIRTK/build/lib/tools/edit-image /home/ref.nii.gz /home/ref.nii.gz -dx 1 -dy 1 -dz 1 ; \
    /bin/MIRTK/build/lib/tools/transform-image /home/data/{date_path}/{date_path}-result/reo-DSVR-output-body.nii.gz \
    /home/data/{date_path}/{date_path}-result/grid-reo-DSVR-output-body.nii.gz -target /home/ref.nii.gz -interp BSpline ; \
    /bin/MIRTK/build/lib/tools/nan /home/data/{date_path}/{date_path}-result/grid-reo-DSVR-output-body.nii.gz 1000000 ; \
    /bin/MIRTK/build/lib/tools/convert-image /home/data/{date_path}/{date_path}-result/grid-reo-DSVR-output-body.nii.gz \
    /home/data/{date_path}/{date_path}-result/grid-reo-DSVR-output-body.nii.gz -short ; \
    chmod 1777 /home/data/{date_path}/{date_path}-result/grid-reo-DSVR-output-body.nii.gz ; \
    suffix=1; \
    while [ -d "/home/data/{date_path}-$suffix" ]; do suffix=$((suffix+1)); done; \
    mv /home/data/{date_path} /home/data/{date_path}-$suffix; \
    mkdir /home/data/{date_path}; \
    chmod 1777 /home/data/{date_path}; ' '''

    # subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
    # subprocess.Popen(command, shell=True)
    print()
    print("--------------------------------------------------------------")
    print()

    currentSeries = 0

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):

        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[..., iImg].transpose((3, 2, 0, 1)), transpose=False)
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # Unused example, as images are grouped by series before being passed into this function now
        # oldHeader.image_series_index = currentSeries

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'),
                                                     'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole'] = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['PYTHON', 'INVERT']
        tmpMeta['WindowCenter'] = '16384'
        tmpMeta['WindowWidth'] = '32768'
        tmpMeta['SequenceDescriptionAdditional'] = 'FIRE'
        tmpMeta['Keep_image_geometry'] = 1
        # tmpMeta['ROI_example']                    = create_example_roi(data.shape)

        # Example for setting colormap
        # tmpMeta['LUTFileName']            = 'MicroDeltaHotMetal.pal'

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]),
                                      "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]),
                                         "{:.18f}".format(oldHeader.phase_dir[1]),
                                         "{:.18f}".format(oldHeader.phase_dir[2])]

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut


# Create an example ROI <3
def create_example_roi(img_size):
    t = np.linspace(0, 2 * np.pi)
    x = 16 * np.power(np.sin(t), 3)
    y = -13 * np.cos(t) + 5 * np.cos(2 * t) + 2 * np.cos(3 * t) + np.cos(4 * t)

    # Place ROI in bottom right of image, offset and scaled to 10% of the image size
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    x = (x * 0.08 * img_size[0]) + 0.82 * img_size[0]
    y = (y * 0.10 * img_size[1]) + 0.80 * img_size[1]

    rgb = (1, 0, 0)  # Red, green, blue color -- normalized to 1
    thickness = 1  # Line thickness
    style = 0  # Line style (0 = solid, 1 = dashed)
    visibility = 1  # Line visibility (0 = false, 1 = true)

    roi = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)
    return roi
