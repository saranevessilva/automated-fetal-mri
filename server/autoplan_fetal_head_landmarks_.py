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


state = {
    "slice_pos": 0,
    "min_slice_pos": 0,
    "first_slice": 1
}


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

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    ncontrasts = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    dim_x = metadata.encoding[0].encodedSpace.matrixSize.x // 2  # oversampling
    dim_y = metadata.encoding[0].encodedSpace.matrixSize.y
    # im = np.zeros((dim_x, dim_y, nslices, ncontrasts), dtype=np.int16)
    im = np.zeros((dim_x, dim_y, nslices), dtype=np.int16)
    # slice_pos = 0
    # min_slice_pos = 0
    # first_slice = 1

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
                    # image = process_image(imgGroup, connection, config, metadata, im,
                    #                       slice_pos, min_slice_pos, first_slice)
                    image = process_image(imgGroup, connection, config, metadata, im, state)

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
            # image = process_image(imgGroup, connection, config, metadata, im, slice_pos, min_slice_pos, first_slice)
            image = process_image(imgGroup, connection, config, metadata, im, state)
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


# def process_image(images, connection, config, metadata, im, slice_pos, min_slice_pos, first_slice):
def process_image(images, connection, config, metadata, im, state):
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

    position = imheader.position
    position = position[0], position[1], position[2]
    slice_dir = imheader.slice_dir
    slice_dir = slice_dir[0], slice_dir[1], slice_dir[2]
    phase_dir = imheader.phase_dir
    phase_dir = phase_dir[0], phase_dir[1], phase_dir[2]
    read_dir = imheader.read_dir
    read_dir = read_dir[0], read_dir[1], read_dir[2]
    print("position ", position, "read_dir", read_dir, "phase_dir ", phase_dir, "slice_dir ", slice_dir)

    # Update state variables directly
    if state["first_slice"] == 1:
        state["min_slice_pos"] = position[1]
        state["first_slice"] = 0
    else:
        if position[1] < state["min_slice_pos"]:
            state["min_slice_pos"] = position[1]

    state["slice_pos"] += position[1]
    pos_z = position[2]
    print("accumulated slice pos", state["slice_pos"])
    print("accumulated position", position[1])
    print("pos_z", pos_z)

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

    slice = imheader.slice
    contrast = imheader.contrast
    repetition = imheader.repetition
    print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

    sform_x = imheader.read_dir
    sform_y = imheader.phase_dir
    sform_z = imheader.slice_dir
    position = imheader.position

    srow_x = (sform_x[0], sform_x[1], sform_x[2])
    srow_y = (sform_y[0], sform_y[1], sform_y[2])
    srow_z = (sform_z[0], sform_z[1], sform_z[2])

    # im = np.squeeze(data)

    # im[:, :, slice, contrast] = np.squeeze(data)  # slice - 1 because 'slice

    # print("Image shape:", im.shape)

    # position = position[0], slice_pos, pos_z
    position = position[0], state["slice_pos"], pos_z

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

    srow_x = (np.round(srow_x, 3))
    srow_y = (np.round(srow_y, 3))
    srow_z = (np.round(srow_z, 3))

    srow_x = (srow_x[0], srow_x[1], srow_x[2])
    srow_y = (srow_y[0], srow_y[1], srow_y[2])
    srow_z = (srow_z[0], srow_z[1], srow_z[2])

    slice = imheader.slice
    repetition = imheader.repetition
    contrast = imheader.contrast
    print("Repetition ", repetition, "Slice ", slice, "Contrast ", contrast)

    # Define the path where the results will be saved
    fetalbody_path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/"
                      + date_path)

    file_path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "-"
                 + timestamp + "-centreofmass.txt")

    # Check if the parent directory exists, if not, create it
    if not os.path.exists(fetalbody_path):
        os.makedirs(fetalbody_path)

    logging.info("Storing each slice into the 3D data buffer...")
    if contrast == 1:
        im[:, :, slice] = np.squeeze(data)

    if slice == nslices-1 and contrast == 1:
        fetal_im_sitk = im
        # fetal_im_sitk = fetal_im_sitk[:, :, 1::2]
        fetal_im_ = fetal_im_sitk

        fetal_im_sitk = sitk.GetImageFromArray(fetal_im_sitk)
        print("Data", im.shape)
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

        print("This is the shape", fetal_im.shape)

        # fetal_im_ = fetal_im[:, :, 1::2]
        # fetal_im_ = fetal_im[..., 0]  # for contrast = 1 & offline testing

        print("..................................................................................")
        print("This is the echo-time we're looking at: ", 1)

        logging.info("Initializing localization network...")

        N_epochs = 100
        I_size = 128
        N_classes = 2

        # # # Prepare arguments

        args = ArgumentsTrainTestLocalisation(epochs=N_epochs,
                                              batch_size=2,
                                              lr=0.002,
                                              crop_height=I_size,
                                              crop_width=I_size,
                                              crop_depth=I_size,
                                              validation_steps=8,
                                              lamda=10,
                                              training=False,
                                              testing=False,
                                              running=True,
                                              root_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron'
                                                       '/python',
                                              csv_dir='/home/sn21/miniconda3/envs/gadgetron/share/gadgetron'
                                                      '/python/files/',
                                              checkpoint_dir='/home/sn21/miniconda3/envs/gadgetron/share'
                                                             '/gadgetron/python/checkpoints/2022-12-16-newest/',
                                              # change to -breech or -young if needed!
                                              train_csv=
                                              'data_localisation_1-label-brain_uterus_train-2022-11-23.csv',
                                              valid_csv=
                                              'data_localisation_1-label-brain_uterus_valid-2022-11-23.csv',
                                              test_csv=
                                              'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                              run_csv=
                                              'data_localisation_1-label-brain_uterus_test-2022-11-23.csv',
                                              # run_input=im_corr2ab,
                                              run_input=fetal_im_,
                                              results_dir='/home/sn21/miniconda3/envs/gadgetron/share'
                                                          '/gadgetron/python/results/',
                                              exp_name='Loc_3D',
                                              task_net='unet_3D',
                                              n_classes=N_classes)

        # path = "/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/2023-09-25/im.nii.gz"
        # image = sitk.GetImageFromArray(im)
        # sitk.WriteImage(image, path)

        # for acquisition in connection:

        args.gpu_ids = [0]

        # RUN with empty masks - to generate new ones (practical application)

        if args.running:
            print("Running")
            # print("im shape ", im_corr2ab.shape)
            logging.info("Starting localization...")
            model = md.LocalisationNetwork3DMultipleLabels(args)
            # Run inference
            ####################
            model.run(args, 1)  # Changing this to 0 avoids the plotting
            logging.info("Localization completed!")

            rotx = 0.0
            roty = 0.0
            rotz = 0.0

            logging.info("Storing motion parameters into variables...")
            xcm = model.x_cm
            ycm = model.y_cm
            zcm = model.z_cm
            logging.info("Motion parameters stored!")

            text = str('CoM: ')
            append_new_line(file_path, text)
            text = str(xcm)
            append_new_line(file_path, text)
            text = str(ycm)
            append_new_line(file_path, text)
            text = str(zcm)
            append_new_line(file_path, text)
            text = str('---------------------------------------------------')
            append_new_line(file_path, text)

            print("centre-of-mass coordinates: ", xcm, ycm, zcm)
            print("Localisation completed.")

        segmentation_volume = model.seg_pr
        image_volume = model.img_gt

        box, expansion_factor, center, offset, side_length, mask, vol, crop = (apply_bounding_box
                                                                               (segmentation_volume,
                                                                                image_volume))

        # Define the path you want to create
        new_directory_seg = fetalbody_path + "/" + timestamp + "-nnUNet_seg/"
        new_directory_pred = fetalbody_path + "/" + timestamp + "-nnUNet_pred/"

        box_path = args.results_dir + date_path

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

        box_im = nib.Nifti1Image(box, np.eye(4))
        nib.save(box_im, box_path + "/" + timestamp + "-nnUNet_seg/FreemaxLandmark_001_0000.nii.gz")
        im_ = nib.Nifti1Image(im, np.eye(4))
        # path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/" + date_path + "/"
        #         + timestamp + "-gadgetron-fetal-brain-localisation-img_initial.nii.gz")
        # nib.save(im_, path)  # SNS! WORK ON CONTRASTS!
        # sitk.WriteImage(im, path)

        # Run Prediction with nnUNet
        # Set the DISPLAY and XAUTHORITY environment variables
        os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
        os.environ["XAUTHORITY"] = '/home/sn21/.Xauthority'

        start_time = time.time()
        terminal_command = (("export nnUNet_raw='/home/sn21/landmark-data/Landmarks/nnUNet_raw'; export"
                             "nnUNet_preprocessed='/home/sn21/landmark-data/Landmarks"
                             "/nnUNet_preprocessed' ; export "
                             "nnUNet_results='/home/sn21/landmark-data/Landmarks/nnUNet_results' ; "
                             "conda activate gadgetron ; nnUNetv2_predict -i ") + box_path + "/" +
                            timestamp + "-nnUNet_seg/ -o " + box_path + "/" + timestamp +
                            "-nnUNet_pred/ -d 080 -c 3d_fullres -f 1")

        subprocess.run(terminal_command, shell=True)
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed Time for Landmark Detection: {elapsed_time} seconds")

        # Define the path where NIfTI images are located
        l_path = os.path.join(box_path, timestamp + "-nnUNet_pred")

        # Use glob to find NIfTI files in the directory
        landmarks_paths = glob.glob(os.path.join(l_path, "*.nii.gz"))
        print(landmarks_paths)

        for landmarks_path in landmarks_paths:
            # Load the image using sitk.ReadImage
            landmark = nib.load(landmarks_path)
            # Get the image data as a NumPy array
            landmark = landmark.get_fdata()
            modified_landmark = np.copy(landmark)

            # Replace label values 2 with 1
            landmark[np.where(landmark == 2.0)] = 3.0
            modified_landmark[np.where(modified_landmark == 2.0)] = 0.0

            # Define a structuring element (a kernel) for morphological operations
            structuring_element = np.ones((3, 3, 3))  # Adjust size if needed

            # Perform morphological closing
            closed_segmentation = ndimage.binary_closing(modified_landmark,
                                                         structure=structuring_element).astype(
                modified_landmark.dtype)

            labelled_segmentation, num_features = ndimage.label(closed_segmentation)
            print("number of features", num_features)
            # region_sizes = np.bincount(labelled.ravel())[1:]

            mask = (landmark == 3.0)
            labelled_segmentation[mask] = 3.0

            eye_1 = (labelled_segmentation == 1.0).astype(int)
            eye_2 = (labelled_segmentation == 2.0).astype(int)
            cereb = (labelled_segmentation == 3.0).astype(int)

            # print("EYE 1", eye_1)

            cm_eye_1 = np.array(ndimage.center_of_mass(eye_1))
            cm_eye_2 = np.array(ndimage.center_of_mass(eye_2))
            cm_cereb = np.array(ndimage.center_of_mass(cereb))

            # cm_eye_1 = cm_eye_1[2], cm_eye_1[1], cm_eye_1[0]
            # cm_eye_2 = cm_eye_2[2], cm_eye_2[1], cm_eye_2[0]
            # cm_cereb = cm_cereb[2], cm_cereb[1], cm_cereb[0]
            print("LANDMARKS", cm_eye_1, cm_eye_2, cm_cereb)

            # # Extract coordinates of non-zero points in the mask
            # points = np.argwhere(segmentation_volume)
            #
            # # Calculate the centroid (you can use a different reference point if needed)
            # centroid = xcm, ycm, zcm
            #
            # # Calculate Euclidean distances from each point to the centroid
            # distances = [euclidean(point, centroid) for point in points]
            #
            # # Find the index of the point with the maximum distance
            # furthest_point_index = np.argmax(distances)
            #
            # # Retrieve the coordinates of the furthest point
            # furthest_point = points[furthest_point_index]
            # print("Furthest Point Coordinates:", furthest_point)

            # Get the current date and time
            current_datetime = datetime.now()

            # Format the date and time as a string
            date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

            # Define the file name with the formatted date and time
            text_file_1 = args.results_dir + date_path + "/" + timestamp + "-nnUNet_pred/" + "com.txt"
            text_file = "/home/sn21/freemax-transfer/Sara/landmarks-interface-autoplan/sara.dvs"

            cm_brain = model.x_cm, model.y_cm, model.z_cm

            # Calculate the scaling factor
            scaling_factor = expansion_factor
            original_dimensions = vol.shape
            scaled_dimensions = crop.shape
            print("original_dimensions", original_dimensions, "scaled_dimensions", scaled_dimensions)

            # Calculate the corresponding coordinates in the scaled (34x34x34) image
            cropped_eye_1_x = cm_eye_1[0] / scaling_factor
            cropped_eye_1_y = cm_eye_1[1] / scaling_factor
            cropped_eye_1_z = cm_eye_1[2] / scaling_factor

            cropped_eye_2_x = cm_eye_2[0] / scaling_factor
            cropped_eye_2_y = cm_eye_2[1] / scaling_factor
            cropped_eye_2_z = cm_eye_2[2] / scaling_factor

            cropped_cereb_x = cm_cereb[0] / scaling_factor
            cropped_cereb_y = cm_cereb[1] / scaling_factor
            cropped_cereb_z = cm_cereb[2] / scaling_factor

            # Print the corresponding coordinates in the scaled image
            print("Coordinates in the scaled image: ({:.2f}, {:.2f}, {:.2f})".format(cropped_eye_1_x,
                                                                                     cropped_eye_1_y,
                                                                                     cropped_eye_1_z))

            # Extract coordinates of non-zero points in the mask
            points = np.argwhere(segmentation_volume)

            # Calculate the centroid (you can use a different reference point if needed)
            centroid = xcm, ycm, zcm

            cm_eye_avg = (cm_eye_1 + cm_eye_2) / 2.0  # this is the anterior landmark

            # Calculate Euclidean distances from each point to the centroid
            distances = [euclidean(point, centroid) for point in points]

            # Compute distances to the average position of the eyes
            dist_to_eyes = [euclidean(point, cm_eye_avg) for point in points]

            # the highest score corresponds to the furthest distance to the centroid and shortest
            # to the eyes
            scores = np.array(distances) - np.array(dist_to_eyes)
            furthest_point_index = np.argmax(scores)

            # Retrieve the coordinates of the furthest point
            furthest_point = points[furthest_point_index]
            print("Furthest Point Coordinates:", furthest_point)

            # Calculate the center of mass in the original 128x128x128 matrix
            cm_eye_1 = (
                int(center[0] - side_length // 2) + cropped_eye_1_x,
                int(center[1] - side_length // 2) + cropped_eye_1_y,
                int(center[2] - side_length // 2) + cropped_eye_1_z
            )
            cm_eye_2 = (
                int(center[0] - side_length // 2) + cropped_eye_2_x,
                int(center[1] - side_length // 2) + cropped_eye_2_y,
                int(center[2] - side_length // 2) + cropped_eye_2_z
            )
            cm_cereb = (
                int(center[0] - side_length // 2) + cropped_cereb_x,
                int(center[1] - side_length // 2) + cropped_cereb_y,
                int(center[2] - side_length // 2) + cropped_cereb_z
            )

            # Dimensions of the padded 128x128x128 image
            padded_dimensions = vol.shape

            # Original dimensions of the image (128x128x60)
            original_dimensions = im.shape

            # Calculate the padding in each dimension
            padding = ((padded_dimensions[0] - original_dimensions[0]) // 2,
                       (padded_dimensions[1] - original_dimensions[1]) // 2,
                       (padded_dimensions[2] - original_dimensions[2]) // 2)

            # Calculate equivalent coordinates in the original 128x128x60 image
            cm_eye_1 = (
                cm_eye_1[0] - padding[0],
                cm_eye_1[1] - padding[1],
                cm_eye_1[2] - padding[2]
            )

            cm_eye_2 = (
                cm_eye_2[0] - padding[0],
                cm_eye_2[1] - padding[1],
                cm_eye_2[2] - padding[2]
            )

            cm_cereb = (
                cm_cereb[0] - padding[0],
                cm_cereb[1] - padding[1],
                cm_cereb[2] - padding[2]
            )

            cm_brain = (
                cm_brain[0] - padding[0],
                cm_brain[1] - padding[1],
                cm_brain[2] - padding[2]
            )

            furthest_point = (
                furthest_point[0] - padding[0],
                furthest_point[1] - padding[1],
                furthest_point[2] - padding[2]
            )

            print("EYE 1", cm_eye_1)
            print("EYE 2", cm_eye_2)
            print("CEREB", cm_cereb)
            print("BRAIN", cm_brain)
            print("FURTHEST BRAIN VOXEL", furthest_point)

            cm_eye_1 = (cm_eye_1[1], cm_eye_1[0], cm_eye_1[2])
            cm_eye_2 = (cm_eye_2[1], cm_eye_2[0], cm_eye_2[2])
            cm_cereb = (cm_cereb[1], cm_cereb[0], cm_cereb[2])
            cm_brain = (cm_brain[1], cm_brain[0], cm_brain[2])
            furthest_point = (furthest_point[1], furthest_point[0], furthest_point[2])

            print("EYE 1", cm_eye_1)
            print("EYE 2", cm_eye_2)
            print("CEREB", cm_cereb)
            print("BRAIN", cm_brain)
            print("FURTHEST BRAIN VOXEL", furthest_point)

            cm_eye_1 = pixdim_x * cm_eye_1[0], pixdim_y * cm_eye_1[1], pixdim_z * cm_eye_1[2]
            cm_eye_2 = pixdim_x * cm_eye_2[0], pixdim_y * cm_eye_2[1], pixdim_z * cm_eye_2[2]
            cm_cereb = pixdim_x * cm_cereb[0], pixdim_y * cm_cereb[1], pixdim_z * cm_cereb[2]
            cm_brain = pixdim_x * cm_brain[0], pixdim_y * cm_brain[1], pixdim_z * cm_brain[2]
            furthest_point = (pixdim_x * furthest_point[0], pixdim_y * furthest_point[1],
                              pixdim_z * furthest_point[2])

            # # # # # # # # # # # # # # # # # # # # # # # # # #
            # position = np.array(position)
            # position = [position[0], position[1], position[2]]
            pos = state["slice_pos"] / (nslices * ncontrasts)  # slice position mid volume
            print("POS", pos)
            print("slice_pos", state["slice_pos"])
            print("nslices", nslices)
            position = (position[0], pos, position[2])

            # lowerleftcorner = ((np.int(enc.encodedSpace.fieldOfView_mm.x/2),
            #                     np.int(enc.encodedSpace.fieldOfView_mm.y/2), np.int(min_slice_pos)))
            centreofimageposition = ((np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.x) / 4,  # oversampling
                                      np.float64(metadata.encoding[0].encodedSpace.fieldOfView_mm.y) / 2,
                                      np.float64(nslices * pixdim_z) / 2))
            print("centreofimageposition", centreofimageposition)

            # position = np.round(position).astype(int)
            position = (position[0], position[1], position[2])
            # position = - position[1], position[2], position[0]
            cm_eye_1 = np.round(cm_eye_1, 3)
            cm_eye_2 = np.round(cm_eye_2, 3)
            cm_cereb = np.round(cm_cereb, 3)
            cm_brain = np.round(cm_brain, 3)
            furthest_point = np.round(furthest_point, 3)

            print("POSITION MM", position)
            print("EYE 1 MM", cm_eye_1)
            print("EYE 2 MM", cm_eye_2)
            print("CEREB MM", cm_cereb)
            print("BRAIN MM", cm_brain)
            print("FURTHEST BRAIN VOXEL MM", furthest_point)

            cm_brain = (cm_brain[0] - centreofimageposition[0],
                        cm_brain[1] - centreofimageposition[1],
                        cm_brain[2] - centreofimageposition[2])

            furthest_point = (furthest_point[0] - centreofimageposition[0],
                              furthest_point[1] - centreofimageposition[1],
                              furthest_point[2] - centreofimageposition[2])

            cm_eye_1 = ((cm_eye_1[0]) - centreofimageposition[0],
                        (cm_eye_1[1]) - centreofimageposition[1],
                        cm_eye_1[2] - centreofimageposition[2])

            cm_eye_2 = ((cm_eye_2[0]) - centreofimageposition[0],
                        (cm_eye_2[1]) - centreofimageposition[1],
                        cm_eye_2[2] - centreofimageposition[2])

            cm_cereb = ((cm_cereb[0]) - centreofimageposition[0],
                        cm_cereb[1] - centreofimageposition[1],
                        cm_cereb[2] - centreofimageposition[2])

            print("centreofimageposition", centreofimageposition)
            print("EYE 1 OFFSET", cm_eye_1)
            print("EYE 2 OFFSET", cm_eye_2)
            print("CEREB OFFSET", cm_cereb)
            print("BRAIN OFFSET", cm_brain)
            print("FURTHEST BRAIN VOXEL OFFSET", furthest_point)

            # x = -ty  # -ty # seems to work
            # y = tz  # tz
            # z = tx  # tx  # seems to work
            #
            # cm_eye_1 = (-cm_eye_1[1], cm_eye_1[2], cm_eye_1[0])
            # cm_eye_2 = (-cm_eye_2[1], cm_eye_2[2], cm_eye_2[0])
            # cm_cereb = (-cm_cereb[1], cm_cereb[2], cm_cereb[0])
            # cm_brain = (-cm_brain[1], cm_brain[2], cm_brain[0])
            # furthest_point = (-furthest_point[1], furthest_point[2], furthest_point[0])

            cm_brain = (cm_brain[0] + position[0],
                        cm_brain[1] + position[1],
                        cm_brain[2] + position[2])

            furthest_point = (furthest_point[0] + position[0],
                              furthest_point[1] + position[1],
                              furthest_point[2] + position[2])

            cm_eye_1 = ((cm_eye_1[0]) + position[0],
                        (cm_eye_1[1]) + position[1],
                        cm_eye_1[2] + position[2])

            cm_eye_2 = ((cm_eye_2[0]) + position[0],
                        (cm_eye_2[1]) + position[1],
                        cm_eye_2[2] + position[2])

            cm_cereb = ((cm_cereb[0]) + position[0],
                        cm_cereb[1] + position[1],
                        cm_cereb[2] + position[2])

            # print("EYE 1 ROT", cm_eye_1)
            # print("EYE 2 ROT", cm_eye_2)
            # print("CEREB ROT", cm_cereb)
            # print("BRAIN ROT", cm_brain)
            # print("FURTHEST BRAIN VOXEL ROT", furthest_point)

            # Find the indices where cm_cereb is NaN
            idx_eye_1 = np.isnan(cm_eye_1)
            # Use numpy.where to replace NaN values with corresponding values from cm_brain
            cm_eye_1 = np.where(idx_eye_1, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_eye_1)

            # Find the indices where cm_cereb is NaN
            idx_eye_2 = np.isnan(cm_eye_2)
            # Use numpy.where to replace NaN values with corresponding values from cm_brain
            cm_eye_2 = np.where(idx_eye_2, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_eye_2)

            # Find the indices where cm_cereb is NaN
            idx_cereb = np.isnan(cm_cereb)
            # Use numpy.where to replace NaN values with corresponding values from cm_brain
            cm_cereb = np.where(idx_cereb, (cm_brain[0], cm_brain[1], cm_brain[2]), cm_cereb)

            cm_eye_1 = (cm_eye_1[0], cm_eye_1[1], cm_eye_1[2])
            cm_eye_2 = (cm_eye_2[0], cm_eye_2[1], cm_eye_2[2])
            cm_cereb = (cm_cereb[0], cm_cereb[1], cm_cereb[2])

            print("EYE 1 ROT", cm_eye_1)
            print("EYE 2 ROT", cm_eye_2)
            print("CEREB ROT", cm_cereb)
            print("BRAIN ROT", cm_brain)
            print("FURTHEST BRAIN VOXEL ROT", furthest_point)

            # transformation = [(srow_x[0], srow_x[1], srow_x[2], srow_x[3]),
            #                   (srow_y[0], srow_y[1], srow_y[2], srow_y[3]),
            #                   (srow_z[0], srow_z[1], srow_z[2], srow_z[3]),
            #                   (0, 0, 0, 1)]

            # Create and write to the text file
            with open(text_file, "w") as file:
                # file.write("This is a text file created on " + date_time_string)
                # file.write("\n" + str('CoM: '))
                file.write("eye1 = " + str(cm_eye_1))
                file.write("\n" + "eye2 = " + str(cm_eye_2))
                file.write("\n" + "cere = " + str(cm_cereb))
                file.write("\n" + "brain = " + str(cm_brain))
                file.write("\n" + "furthest = " + str(furthest_point))
                file.write("\n" + "position = " + str(position))
                # file.write("\n" + "centreofimageposition = " + str(centreofimageposition))
                file.write("\n" + "srow_x = " + str(srow_x))
                file.write("\n" + "srow_y = " + str(srow_y))
                file.write("\n" + "srow_z = " + str(srow_z))

            with open(text_file_1, "w") as file:
                # file.write("This is a text file created on " + date_time_string)
                # file.write("\n" + str('CoM: '))
                file.write("eye1 = " + str(cm_eye_1))
                file.write("\n" + "eye2 = " + str(cm_eye_2))
                file.write("\n" + "cere = " + str(cm_cereb))
                file.write("\n" + "brain = " + str(cm_brain))
                file.write("\n" + "furthest = " + str(furthest_point))
                file.write("\n" + "position = " + str(position))
                file.write("\n" + "srow_x = " + str(srow_x))
                file.write("\n" + "srow_y = " + str(srow_y))
                file.write("\n" + "srow_z = " + str(srow_z))

            print(f"Text file '{text_file}' has been created.")

            # Convert the modified NumPy array back to a SimpleITK image
            # modified_image = sitk.GetImageFromArray(labelled_segmentation)
            # modified_image = labelled_segmentation

            # Save the modified image with the same file path
            output_image_path = landmarks_path.replace(".nii.gz", "_3-labels.nii.gz")

            # sitk.WriteImage(modified_image, output_image_path)
            modified_image = nib.Nifti1Image(labelled_segmentation, np.eye(4))
            nib.save(modified_image, output_image_path)

        else:
            print("This is slice ", slice)

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
