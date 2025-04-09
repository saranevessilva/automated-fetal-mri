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

import src.utils as utils
from src.utils import ArgumentsTrainTestLocalisation, plot_losses_train
from src import networks as md
from src.boundingbox import calculate_expanded_bounding_box, apply_bounding_box
# import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

# standard libraries
from matplotlib import pyplot as plt

import os

# import numpy as np
# import matplotlib as plt
import math
import pandas as pd
import scipy
from skimage import io

# tools for volume rendering and dynamic visualisation
from skimage import measure
import plotly
import plotly.graph_objs as go
import plotly.express as px
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

# NiBabel and NiLean tools for I/O and visualisation of .nii images
# import nibabel as nib
from nilearn.plotting import view_img, plot_glass_brain, plot_anat, plot_img, plot_roi
from nilearn.image import resample_to_img, resample_img
from scipy.ndimage import zoom
from plotly.subplots import make_subplots
from scipy.stats import norm

import img2pdf
from PIL import Image

import PyPDF2

# switch off warning messages
import warnings

warnings.filterwarnings("ignore")

import plotly.io as pio
# import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.express as px
import skimage.io as sio
# import numpy as np
from numpy import pi, sin, cos

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
    # global timestamp
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

    pixdim_x = (metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x) / 2
    pixdim_y = (metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[0].encodedSpace.matrixSize.y)
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

    volumetry_path = debugFolder + "/" + date_path

    # Check if the parent directory exists, if not, create it
    if not os.path.exists(volumetry_path):
        os.makedirs(volumetry_path)

    # Define the path you want to create
    new_directory_seg = volumetry_path + "/" + timestamp + "-nnUNet_seg-volumetry/"
    new_directory_pred = volumetry_path + "/" + timestamp + "-nnUNet_pred-volumetry/"

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
                    volumetry_path + "/" + timestamp + "-output.nii.gz")

    sitk.WriteImage(fetal_im_sitk,
                    volumetry_path + "/" + timestamp + "-nnUNet_seg-volumetry/Volumetry_001_0000.nii"
                                                       ".gz")

    print("The images have been saved!")
    # sitk.WriteImage(im, path)

    path = "/opt/code/automated-fetal-mri/whole-uterus-segmentation-reporting.py"

    result = subprocess.run(
        f"python3 {path}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True
    )

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
