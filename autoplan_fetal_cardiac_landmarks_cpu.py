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

import gadgetron
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

import src.utils as utils
from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md
from src.boundingbox import calculate_expanded_bounding_box, apply_bounding_box
from numpy.fft import fftshift, ifftshift, fftn, ifftn

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def adjust_contrast(image_array, mid_intensity, target_y):
    # Calculate the intensity range
    max_intensity = np.abs(np.max(image_array))
    min_intensity = np.abs(np.min(image_array))
    intensity_range = max_intensity - min_intensity

    # Precompute constant values
    ratio1 = (target_y - 0) / (mid_intensity - min_intensity)
    ratio2 = (1 - target_y) / (max_intensity - mid_intensity)

    # Apply the transformation to the entire array
    adjusted_array = np.where(image_array < mid_intensity,
                              (image_array - min_intensity) * ratio1,
                              (image_array - mid_intensity) * ratio2 + target_y)

    # Adjust the intensity range to match the original range
    adjusted_array = (adjusted_array - np.min(adjusted_array)) * (
            intensity_range / (np.max(adjusted_array) - np.min(adjusted_array))) + min_intensity

    return adjusted_array


def calculate_bounding_box_nonzero(seg_data, expansion_factor=0.0):
    nonzero_indices = np.nonzero(seg_data)
    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)
    expansion = np.round((max_indices - min_indices + 1) * expansion_factor).astype(int)
    lower_bound = np.maximum(min_indices - expansion, 0)
    upper_bound = np.minimum(max_indices + expansion, np.array(seg_data.shape) - 1)
    return lower_bound, upper_bound


def apply_bounding_box(segmentation_file, image_file, output_file, expansion_factor=0.0):
    # Load segmentation image
    segmentation_img = nib.load(segmentation_file)
    segmentation_data = segmentation_img.get_fdata()
    # Calculate bounding box based on nonzero values
    lower_bound, upper_bound = calculate_bounding_box_nonzero(segmentation_data, expansion_factor)
    # Load the corresponding 3D image
    image_img = nib.load(image_file)
    image_data = image_img.get_fdata()
    # Crop the image using the bounding box
    cropped = image_data[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]]
    # Save the cropped result to the output file
    print("lower bound:", lower_bound, "upper bound:", upper_bound)
    output_img = nib.Nifti1Image(cropped, image_img.affine)
    nib.save(output_img, output_file)

    return cropped, lower_bound, upper_buond, expansion_factor


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


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

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, connection, config, metadata)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d",
                                 item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images
                # with unknown type)
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
            image = process_image(imgGroup, connection, config, metadata)
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


def process_image(images, connection, config, metadata):
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images),
                  ismrmrd.get_dtype_from_data_type(images[0].data_type))

    date_path = datetime.today().strftime("%Y-%m-%d")
    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data for img in images])
    head = [img.getHead() for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    imheader = head[0]

    pixdim_x = (metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x)
    pixdim_y = metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[0].encodedSpace.matrixSize.y
    pixdim_z = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    print("pixdims", pixdim_x, pixdim_y, pixdim_z)
    # pixdim_z = 2.4

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))
    print("reformatted data", data.shape)

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    # data *= 32767/data.max()
    # data = np.around(data)
    # data = data.astype(np.int16)

    # Invert image contrast
    # data = 32767-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    currentSeries = 0

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1

    im = np.squeeze(data)

    # im_path = \
    #     ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/"
    #      "cardiac/2024-05-28/12-30-27-output.nii.gz")  # testing only! Sara!
    #
    # im = nib.load(im_path)
    # im = im.get_fdata()

    # slice_fov = 0
    # min_slice_pos = 0
    # first_slice = 1

    print("Image Shape:", im.shape)

    position = imheader.position
    position = position[0], position[1], position[2]

    # this is for ascending order - create for descending / interleaved slices
    slice_thickness = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    print("slice_thickness", slice_thickness)
    slice_pos = position[1] - ((nslices / 2) - 0.5) * slice_thickness  # mid-slice position
    # pos_z = patient_table_position[2] + position[2]
    pos_z = position[2]
    print("mid slice pos", slice_pos)
    print("last position", position[1])
    print("pos_z", pos_z)

    position = position[0], slice_pos, pos_z

    sform_x = imheader.slice_dir
    sform_y = imheader.phase_dir
    sform_z = imheader.read_dir

    srow_x = (sform_x[0], sform_x[1], sform_x[2])
    srow_y = (sform_y[0], sform_y[1], sform_y[2])
    srow_z = (sform_z[0], sform_z[1], sform_z[2])

    srow_x = (np.round(srow_x, 3))
    srow_y = (np.round(srow_y, 3))
    srow_z = (np.round(srow_z, 3))

    srow_x = (srow_x[0], srow_x[1], srow_x[2])
    srow_y = (srow_y[0], srow_y[1], srow_y[2])
    srow_z = (srow_z[0], srow_z[1], srow_z[2])

    # patient_table_position = (imheader.patient_table_position[0], imheader.patient_table_position[1],
    #                           imheader.patient_table_position[2])
    # print("position ", position, "read_dir", read_dir, "phase_dir ", phase_dir, "slice_dir ", slice_dir)
    # print("patient table position", patient_table_position)

    slice = imheader.slice
    repetition = imheader.repetition
    contrast = imheader.contrast
    print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

    # Define the path where the results will be saved
    fetalbody_path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/cardiac/"
                      + date_path)

    # Check if the parent directory exists, if not, create it
    if not os.path.exists(fetalbody_path):
        os.makedirs(fetalbody_path)

    # Define the path you want to create
    new_directory_seg = fetalbody_path + "/" + timestamp + "-nnUNet_seg-fetalbody/"
    new_directory_pred = fetalbody_path + "/" + timestamp + "-nnUNet_pred-fetalbody/"

    # Check if the directory already exists
    if not os.path.exists(new_directory_seg):
        # If it doesn't exist, create it
        os.mkdir(new_directory_seg)
    else:
        # If it already exists, handle it accordingly (maybe log a message or take alternative action)
        print("Directory already exists:", new_directory_seg)

    # Check if the directory already exists
    if not os.path.exists(new_directory_pred):
        # If it doesn't exist, create it
        os.mkdir(new_directory_pred)
    else:
        # If it already exists, handle it accordingly (maybe log a message or take alternative action)
        print("Directory already exists:", new_directory_pred)

    fetal_im_sitk = im

    fetal_im_sitk = sitk.GetImageFromArray(fetal_im_sitk)
    voxel_sizes = (pixdim_z, pixdim_y, pixdim_x)  # Define the desired voxel sizes in millimeters
    srows = srow_x[0], srow_x[1], srow_x[2], srow_y[0], srow_y[1], srow_y[2], srow_z[0], srow_z[1], srow_z[2]
    print("VOXEL SIZE", voxel_sizes)
    fetal_im_sitk.SetSpacing(voxel_sizes)
    fetal_im_sitk.SetDirection(srows)
    print("New spacing has been set!")
    fetal_im = sitk.GetArrayFromImage(fetal_im_sitk)

    fetal_im_sitk = sitk.PermuteAxes(fetal_im_sitk, [1, 2, 0])
    print("Size after transposition:", fetal_im_sitk.GetSize())

    sitk.WriteImage(fetal_im_sitk,
                    fetalbody_path + "/" + timestamp + "-output.nii.gz")

    sitk.WriteImage(fetal_im_sitk,
                    fetalbody_path + "/" + timestamp + "-nnUNet_seg-fetalbody/FetalBody_001_0000.nii.gz")

    print("The images have been saved!")
    # sitk.WriteImage(im, path)

    # Run prediction with nnUNet
    # Set the DISPLAY and XAUTHORITY environment variables
    os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
    os.environ["XAUTHORITY"] = '/home/sn21/.Xauthority'

    # Record the start time
    start_time = time.time()

    # timestamp = "18-14-01"

    # Define the terminal command for prediction
    terminal_command = (("export nnUNet_raw='/home/sn21/landmark-data/FetalBody/nnUNet_raw'; export "
                         "nnUNet_preprocessed='/home/sn21/landmark-data/FetalBody/nnUNet_preprocessed'; "
                         "export nnUNet_results='/home/sn21/landmark-data/FetalBody/nnUNet_results'; "
                         "conda activate gadgetron; nnUNetv2_predict -i ") + fetalbody_path + "/"
                        + timestamp + "-nnUNet_seg-fetalbody/ -o " + fetalbody_path + "/" + timestamp
                        + "-nnUNet_pred-fetalbody/ -d 081 -c 3d_fullres -f 1")

    # Run the terminal command
    subprocess.run(terminal_command, shell=True)  # Sara!

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for fetal body localisation: {elapsed_time} seconds")

    # Load the segmentation and image volumes

    segmentation_filename = os.path.join(fetalbody_path, timestamp + "-nnUNet_pred-fetalbody",
                                         "FetalBody_001.nii.gz")

    segmentation_volume = sitk.ReadImage(segmentation_filename)

    # Removing all smaller segmentations - keep only thorax
    # Compute the size of each connected component
    label_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    label_stats_filter.Execute(segmentation_volume)

    # Get a list of all labels and their sizes
    labels = label_stats_filter.GetLabels()
    label_sizes = [label_stats_filter.GetNumberOfPixels(label) for label in labels]
    print("LABEL SIZES", label_sizes)

    # Identify the label corresponding to the largest segmentation
    largest_label = labels[label_sizes.index(max(label_sizes))]

    # Create a binary mask preserving only the largest segmentation
    largest_segmentation_mask = sitk.BinaryThreshold(segmentation_volume, lowerThreshold=largest_label,
                                                     upperThreshold=largest_label)

    # Apply the binary mask to the original segmentation volume
    segmentation_volume = sitk.Mask(segmentation_volume, largest_segmentation_mask)

    # Save the modified segmentation volume
    output_segmentation_filename = os.path.join(fetalbody_path, timestamp + "-nnUNet_pred-fetalbody",
                                                "FetalBody_001.nii.gz")
    # sitk.WriteImage(segmentation_volume, output_segmentation_filename)  # Sara!

    # Convert the image to a numpy array
    segmentation_volume = sitk.GetArrayFromImage(segmentation_volume)

    # Sara!
    image_filename = os.path.join(fetalbody_path, timestamp + "-nnUNet_seg-fetalbody",
                                  "FetalBody_001_0000.nii.gz")
    # image_volume = nib.load(image_filename).get_fdata()
    # Load the image using SimpleITK
    image_volume = sitk.ReadImage(image_filename)

    # Convert the image to a numpy array
    image_volume = sitk.GetArrayFromImage(image_volume)

    # Define the output filename
    output_filename = os.path.join(fetalbody_path,
                                   timestamp + "-gadgetron-fetalbody-localisation-img_cropped.nii.gz")

    # Calculate bounding box based on nonzero values
    lower_bound, upper_bound = calculate_bounding_box_nonzero(segmentation_volume, expansion_factor=0.0)

    print("lower bound:", lower_bound, "upper bound:", upper_bound)

    # Crop the image using the bounding box
    cropped = image_volume[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1],
              lower_bound[2]:upper_bound[2]]

    lower_left_corner = lower_bound[2], lower_bound[1], lower_bound[0]
    # this is used for finding the coordinates of the landmarks in the uncropped image

    cropped = sitk.GetImageFromArray(cropped)
    # Set the spacing
    voxel_sizes = (pixdim_y, pixdim_x, pixdim_z)  # Define the desired voxel sizes in millimeters
    cropped.SetSpacing(voxel_sizes)

    # Save the SimpleITK image
    sitk.WriteImage(cropped,
                    fetalbody_path + "/" + timestamp +
                    "-gadgetron-fetalbody-localisation-img_cropped.nii.gz")  # Sara!

    print("..................................................................................")
    print("Starting landmark detection...")

    landmarks_path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/cardiac/"
                      + date_path)
    os.mkdir(fetalbody_path + "/" + timestamp + "-nnUNet_seg-landmarks/")
    os.mkdir(fetalbody_path + "/" + timestamp + "-nnUNet_pred-landmarks/")

    fetal_im_sitk.SetSpacing(voxel_sizes)
    sitk.WriteImage(cropped, fetalbody_path + "/" + timestamp
                    + "-nnUNet_seg-landmarks/CardiacLandmarks_001_0000.nii.gz")  # Sara!

    # nib.save(fetalbody_im, fetalbody_path + "/" + timestamp + "-nnUNet_seg-landmarks/"
    #                                                           "CardiacLandmarks_001_0000.nii.gz")

    start_time = time.time()
    # terminal_command = (("export nnUNet_raw='/home/sn21/landmark-data/CardiacLandmarks/nnUNet_raw'; "
    #                      "export nnUNet_preprocessed='/home/sn21/landmark-data/CardiacLandmarks"
    #                      "/nnUNet_preprocessed' ; export "
    #                      "nnUNet_results='/home/sn21/landmark-data/CardiacLandmarks/nnUNet_results' ; "
    #                      "conda activate gadgetron ; nnUNetv2_predict -i ") + landmarks_path + "/" +
    #                     timestamp + "-nnUNet_seg-landmarks/ -o " + landmarks_path + "/" + timestamp +
    #                     "-nnUNet_pred-landmarks/ -d 082 -c 3d_fullres -f 1")

    # new network
    terminal_command = (("export nnUNet_raw='/home/sn21/landmark-data/FetalCardiacLandmarks/nnUNet_raw'; "
                         "export nnUNet_preprocessed='/home/sn21/landmark-data/FetalCardiacLandmarks"
                         "/nnUNet_preprocessed' ; export "
                         "nnUNet_results='/home/sn21/landmark-data/FetalCardiacLandmarks/nnUNet_results' ; "
                         "conda activate gadgetron ; nnUNetv2_predict -i ") + landmarks_path + "/" +
                        timestamp + "-nnUNet_seg-landmarks/ -o " + landmarks_path + "/" + timestamp +
                        "-nnUNet_pred-landmarks/ -d 083 -c 3d_fullres -f 1")

    subprocess.run(terminal_command, shell=True)  # Sara!
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for cardiac landmark detection: {elapsed_time} seconds")

    # Define the path where NIfTI images are located
    landmarks_path = os.path.join(fetalbody_path, timestamp + "-nnUNet_pred-landmarks/CardiacLandmarks_001.nii.gz")

    print(landmarks_path)

    # for landmarks_path in landmarks_paths:
    # Load the image using sitk.ReadImage
    landmark = nib.load(landmarks_path)
    # Get the image data as a NumPy array
    landmark = landmark.get_fdata()

    # dao = (landmark == 1.0).astype(int)
    # dao_diaph = (landmark == 2.0).astype(int)
    # uv = (landmark == 3.0).astype(int)
    # apices = (landmark == 4.0).astype(int)
    # heart_liver = (landmark == 5.0).astype(int)

    # new network
    uv = (landmark == 1.0).astype(int)
    dao = (landmark == 2.0).astype(int)

    nonzero_indices = np.nonzero(dao)

    # Get the minimum and maximum y values - TEST THIS IN MORE FETUSES! Sara!
    # draw a straight line/vector through the vessel here and check start and end points (not always y)
    dao_min_y = np.min(nonzero_indices[1])
    dao_max_y = np.max(nonzero_indices[1])

    # Find the corresponding x and z coordinates for the minimum and maximum y values
    dao_start = (nonzero_indices[0][0], dao_max_y, nonzero_indices[2][0])
    dao_end = (nonzero_indices[0][-1], dao_min_y, nonzero_indices[2][-1])

    print("Beginning coordinates of the descending aorta (x, y, z):", dao_start)
    print("Ending coordinates of the descending aorta (x, y, z):", dao_end)

    dao_start_x, dao_start_y, dao_start_z = dao_start
    dao_start = (dao_start[0] + lower_left_corner[0], dao_start[1] + lower_left_corner[1],
                 dao_start[2] + lower_left_corner[2])

    dao_end_x, dao_end_y, dao_end_z = dao_end
    dao_end = (dao_end[0] + lower_left_corner[0], dao_end[1] + lower_left_corner[1],
               dao_end[2] + lower_left_corner[2])

    dao_centre_x = (dao_start_x + dao_end_x) // 2
    dao_centre_y = (dao_start_y + dao_end_y) // 2
    dao_centre_z = (dao_start_z + dao_end_z) // 2

    dao_centre = (dao_centre_x + lower_left_corner[0], dao_centre_y + lower_left_corner[1],
                  dao_centre_z + lower_left_corner[2])

    # Sara!
    # dao_start_x = 163
    # dao_start_y = 48
    # dao_start_z = 3
    #
    # dao_end_x = 105
    # dao_end_y = 181
    # dao_end_z = 5
    #
    # dao_centre_x = (dao_start_x + dao_end_x) // 2
    # dao_centre_y = (dao_start_y + dao_end_y) // 2
    # dao_centre_z = (dao_start_z + dao_end_z) // 2

    # dao_start = dao_start_x, dao_start_y, dao_start_z
    # dao_end = dao_end_x, dao_end_y, dao_end_z
    # dao_centre = dao_centre_x, dao_centre_y, dao_centre_z

    # Print the center y value and its corresponding coordinates
    print("Coordinates of centre y value (x, y, z):", dao_centre)

    nonzero_indices = np.nonzero(uv)

    # Get the minimum and maximum y values
    uv_min_y = np.min(nonzero_indices[1])
    uv_max_y = np.max(nonzero_indices[1])

    # Find the corresponding x and z coordinates for the minimum and maximum y values
    uv_start = (nonzero_indices[0][0], uv_min_y, nonzero_indices[2][0])
    uv_end = (nonzero_indices[0][-1], uv_max_y, nonzero_indices[2][-1])

    print("Beginning coordinates of the umbilical vein (x, y, z):", uv_start)
    print("Ending coordinates of the umbilical vein (x, y, z):", uv_end)

    uv_start_x, uv_start_y, uv_start_z = uv_start
    uv_start = (uv_start[0] + lower_left_corner[0], uv_start[1] + lower_left_corner[1],
                uv_start[2] + lower_left_corner[2])

    uv_end_x, uv_end_y, uv_end_z = uv_end
    uv_end = (uv_end[0] + lower_left_corner[0], uv_end[1] + lower_left_corner[1],
              uv_end[2] + lower_left_corner[2])

    uv_centre_x = (uv_start_x + uv_end_x) // 2
    uv_centre_y = (uv_start_y + uv_end_y) // 2
    uv_centre_z = (uv_start_z + uv_end_z) // 2

    uv_centre = (uv_centre_x + lower_left_corner[0], uv_centre_y + lower_left_corner[1],
                 uv_centre_z + lower_left_corner[2])

    # Sara!
    # uv_start_x = 125
    # uv_start_y = 115
    # uv_start_z = 3
    #
    # uv_end_x = 205
    # uv_end_y = 115
    # uv_end_z = 3
    #
    # uv_centre_x = (uv_start_x + uv_end_x) // 2
    # uv_centre_y = (uv_start_y + uv_end_y) // 2
    # uv_centre_z = (uv_start_z + uv_end_z) // 2
    #
    # uv_start = uv_start_x, uv_start_y, uv_start_z
    # uv_end = uv_end_x, uv_end_y, uv_end_z
    # uv_centre = uv_centre_x, uv_centre_y, uv_centre_z

    # at this point, we have the landmarks coordinates relative to the original/uncropped image

    print("lowerleftcorner", lower_left_corner)

    # Print the center y value and its corresponding coordinates
    print("Coordinates of centre y value (x, y, z):", uv_centre)

    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the file name with the formatted date and time
    text_file_1 = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/cardiac/" +
                   date_path + "/" + timestamp + "-com_cardiac.txt")
    text_file = "/home/sn21/freemax-transfer/Sara/landmarks-interface-autoplan/sara_cardiac.dvs"

    # # Get the dimensions of the cropped image
    # fetal_im_sitk = sitk.GetArrayFromImage(fetal_im_sitk)
    # cropped = sitk.GetArrayFromImage(cropped)

    print("Upper bound", upper_bound, "Lower bound", lower_bound)

    print("DAO START", dao_start)
    print("DAO END", dao_end)
    print("UV START", uv_start)
    print("UV END", uv_end)
    print("DAO CENTRE", dao_centre)
    print("UV CENTRE", uv_centre)
    print("POSITION", position)

    dao_start = pixdim_x * dao_start[0], pixdim_y * dao_start[1], pixdim_z * dao_start[2]
    dao_end = pixdim_x * dao_end[0], pixdim_y * dao_end[1], pixdim_z * dao_end[2]
    uv_start = pixdim_x * uv_start[0], pixdim_y * uv_start[1], pixdim_z * uv_start[2]
    uv_end = pixdim_x * uv_end[0], pixdim_y * uv_end[1], pixdim_z * uv_end[2]  # JAV
    dao_centre = pixdim_x * dao_centre[0], pixdim_y * dao_centre[1], pixdim_z * dao_centre[2]
    uv_centre = pixdim_x * uv_centre[0], pixdim_y * uv_centre[1], pixdim_z * uv_centre[2]
    # here, we have the landmarks coordinates in mm (check if pixel dimensions are correct)

    print("pixel dimensions", pixdim_x, pixdim_y, pixdim_z)

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    ncontrasts = 1

    centreofimageposition = ((np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.x) / 4,  # OS
                              np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.y) / 2,
                              np.float64(nslices * pixdim_z) / 2))  # JAV

    print("centreofimageposition", centreofimageposition)
    print("fieldOfView_mm.x", np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.x),
          "fieldOfView_mm.y", np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.y),
          "fieldOfView_mm.z", np.float64(nslices * pixdim_z))

    position = np.round(position, 3)
    dao_start = np.round(dao_start, 3)
    dao_end = np.round(dao_end, 3)
    uv_start = np.round(uv_start, 3)
    uv_end = np.round(uv_end, 3)
    dao_centre = np.round(dao_centre, 3)
    uv_centre = np.round(uv_centre, 3)

    print("POSITION MM", position)
    print("DAO START MM", dao_start)
    print("DAO END MM", dao_end)
    print("UV START MM", uv_start)
    print("UV END MM", uv_end)
    print("DAO CENTRE MM", dao_centre)
    print("UV CENTRE MM", uv_centre)

    dao_start = (dao_start[0] - centreofimageposition[0],
                 dao_start[1] - centreofimageposition[1],
                 dao_start[2] - centreofimageposition[2])  # JAV

    dao_end = (dao_end[0] - centreofimageposition[0],
               dao_end[1] - centreofimageposition[1],
               dao_end[2] - centreofimageposition[2])

    uv_start = ((uv_start[0]) - centreofimageposition[0],
                (uv_start[1]) - centreofimageposition[1],
                uv_start[2] - centreofimageposition[2])

    uv_end = ((uv_end[0]) - centreofimageposition[0],
              (uv_end[1]) - centreofimageposition[1],
              uv_end[2] - centreofimageposition[2])

    dao_centre = ((dao_centre[0]) - centreofimageposition[0],
                  dao_centre[1] - centreofimageposition[1],
                  dao_centre[2] - centreofimageposition[2])

    uv_centre = ((uv_centre[0]) - centreofimageposition[0],
                 uv_centre[1] - centreofimageposition[1],
                 uv_centre[2] - centreofimageposition[2])

    print("centreofimageposition", centreofimageposition)
    print("DAO START OFFSET", dao_start)
    print("DAO END OFFSET", dao_end)
    print("UV START OFFSET", uv_start)
    print("UV END OFFSET", uv_end)
    print("DAO CENTRE OFFSET", dao_centre)
    print("UV CENTRE OFFSET", uv_centre)

    # x = -ty  # -ty # seems to work
    # y = tz  # tz
    # z = tx  # tx  # seems to work

    # translation of image space to scanner space
    dao_start = (dao_start[0], dao_start[2], dao_start[1])
    dao_end = (dao_end[0], dao_end[2], dao_end[1])
    uv_start = (uv_start[0], uv_start[2], uv_start[1])
    uv_end = (uv_end[0], uv_end[2], uv_end[1])
    dao_centre = (dao_centre[0], dao_centre[2], dao_centre[1])
    uv_centre = (uv_centre[0], uv_centre[2], uv_centre[1])

    dao_start = (dao_start[0] + position[0],
                 dao_start[1] + position[1],
                 dao_start[2] + position[2])

    dao_end = (dao_end[0] + position[0],
               dao_end[1] + position[1],
               dao_end[2] + position[2])

    uv_start = ((uv_start[0]) + position[0],
                (uv_start[1]) + position[1],
                uv_start[2] + position[2])

    uv_end = ((uv_end[0]) + position[0],
              (uv_end[1]) + position[1],
              uv_end[2] + position[2])

    dao_centre = ((dao_centre[0]) + position[0],
                  dao_centre[1] + position[1],
                  dao_centre[2] + position[2])

    uv_centre = ((uv_centre[0]) + position[0],
                 uv_centre[1] + position[1],
                 uv_centre[2] + position[2])

    # WRITE AN IF STATEMENT HERE. Sara!
    # Find the indices where cm_cereb is NaN
    idx_dao_start = np.isnan(dao_start)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    dao_start = np.where(idx_dao_start, (centreofimageposition[0], centreofimageposition[1],
                                         centreofimageposition[2]), dao_start)

    # Find the indices where cm_cereb is NaN
    idx_dao_end = np.isnan(dao_end)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    dao_end = np.where(idx_dao_end, (centreofimageposition[0], centreofimageposition[1],
                                     centreofimageposition[2]), dao_end)

    # Find the indices where cm_cereb is NaN
    idx_uv_start = np.isnan(uv_start)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    uv_start = np.where(idx_uv_start, (centreofimageposition[0], centreofimageposition[1],
                                       centreofimageposition[2]), uv_start)

    # Find the indices where cm_cereb is NaN
    idx_uv_end = np.isnan(uv_end)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    uv_end = np.where(idx_uv_end, (centreofimageposition[0], centreofimageposition[1],
                                   centreofimageposition[2]), uv_end)

    # Find the indices where cm_cereb is NaN
    idx_dao_centre = np.isnan(dao_centre)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    dao_centre = np.where(idx_dao_centre, (centreofimageposition[0], centreofimageposition[1],
                                           centreofimageposition[2]), dao_centre)

    # Find the indices where cm_cereb is NaN
    idx_uv_centre = np.isnan(uv_centre)
    # Use numpy.where to replace NaN values with corresponding values from cm_brain
    uv_centre = np.where(idx_uv_centre, (centreofimageposition[0], centreofimageposition[1],
                                         centreofimageposition[2]), uv_centre)

    dao_start = (dao_start[0], dao_start[1], -dao_start[2])
    dao_end = (dao_end[0], dao_end[1], -dao_end[2])
    uv_start = (uv_start[0], uv_start[1], -uv_start[2])
    uv_end = (uv_end[0], uv_end[1], -uv_end[2])
    dao_centre = (dao_centre[0], dao_centre[1], -dao_centre[2])
    uv_centre = (uv_centre[0], uv_centre[1], -uv_centre[2])
    position = (position[0], position[1], position[2])

    print("DAO START ROT", dao_start)
    print("DAO END ROT", dao_end)
    print("UV START ROT", uv_start)
    print("UV END ROT", uv_end)
    print("DAO CENTRE ROT", dao_centre)
    print("UV CENTRE ROT", uv_centre)
    print("POSITION ROT", position)

    # transformation = [(srow_x[0], srow_x[1], srow_x[2], srow_x[3]),
    #                   (srow_y[0], srow_y[1], srow_y[2], srow_y[3]),
    #                   (srow_z[0], srow_z[1], srow_z[2], srow_z[3]),
    #                   (0, 0, 0, 1)]

    # Create and write to the text file
    with open(text_file, "w") as file:
        # file.write("This is a text file created on " + date_time_string)
        # file.write("\n" + str('CoM: '))
        file.write("daostart = " + str(dao_start))
        file.write("\n" + "daoend = " + str(dao_end))
        file.write("\n" + "uvstart = " + str(uv_start))
        file.write("\n" + "uvend = " + str(uv_end))
        file.write("\n" + "daocentre = " + str(dao_centre))
        file.write("\n" + "uvcentre = " + str(uv_centre))
        file.write("\n" + "position = " + str(position))
        # file.write("\n" + "centreofimageposition = " + str(centreofimageposition))
        # file.write("\n" + "srow_x = " + str(srow_x))
        # file.write("\n" + "srow_y = " + str(srow_y))
        # file.write("\n" + "srow_z = " + str(srow_z))

    with open(text_file_1, "w") as file:
        # file.write("This is a text file created on " + date_time_string)
        # file.write("\n" + str('CoM: '))
        file.write("daostart = " + str(dao_start))
        file.write("\n" + "daoend = " + str(dao_end))
        file.write("\n" + "uvstart = " + str(uv_start))
        file.write("\n" + "uvend = " + str(uv_end))
        file.write("\n" + "daocentre = " + str(dao_centre))
        file.write("\n" + "uvcentre = " + str(uv_centre))
        file.write("\n" + "position = " + str(position))
        # file.write("\n" + "centreofimageposition = " + str(centreofimageposition))
        # file.write("\n" + "srow_x = " + str(srow_x))
        # file.write("\n" + "srow_y = " + str(srow_y))
        # file.write("\n" + "srow_z = " + str(srow_z))

    print(f"Text file '{text_file}' has been created.")

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]

    for iImg in range(data.shape[-1]):
        # print("iImg", iImg)
        # print("range", data.shape[-1])

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
