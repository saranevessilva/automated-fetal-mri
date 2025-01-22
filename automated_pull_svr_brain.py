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

from sympy.codegen.ast import continue_

import mrdhelper
import constants
import time
from time import perf_counter
from datetime import datetime
import nibabel as nib
import SimpleITK as sitk

import subprocess

import torch
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, UNet, AttentionUnet
from scipy.ndimage import zoom
import skimage

from scipy import ndimage
from skimage.measure import label, regionprops

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Folder for debug output files
# debugFolder = "/tmp/share/debug"
date_path = datetime.today().strftime("%Y-%m-%d")
debugFolder = "/home/sn21/data/t2-stacks/" + date_path



def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        # logging.info("ECHOES", metadata.sequenceParameters)
        logging.info(
            "First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s "
            "x %s)",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

        # voxel_dims = ((metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[
        # 0].encodedSpace.matrixSize.x), (metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[
        # 0].encodedSpace.matrixSize.y), (metadata.encoding[0].encodedSpace.fieldOfView_mm.z / metadata.encoding[
        # 0].encodedSpace.matrixSize.z)) print("VOXEL DIMS", voxel_dims)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)
        # logging.info("ECHOES", metadata.sequenceParameters)

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    dim_x = metadata.encoding[0].encodedSpace.matrixSize.x // 2
    dim_y = metadata.encoding[0].encodedSpace.matrixSize.y
    im = np.zeros((dim_x, dim_y, nslices), dtype=np.int16)

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

                # When these criteria are met, run process_raw() on the accumulated
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
                # When these criteria are met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d",
                                 item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata, im)
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
            image = process_image(imgGroup, connection, config, metadata, im)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()


def process_image(images, connection, config, metadata, im):
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    imheader = head[0]

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

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
    #
    # # Invert image contrast
    # # data = 32767-data
    # data = np.abs(data)
    # data = data.astype(np.int16)
    # np.save(debugFolder + "/" + "imgInverted.npy", data)

    currentSeries = 0

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1

    currentSeries = 0

    slice = imheader.slice

    # online testing

    # im[:, :, slice] = np.squeeze(data)  # slice - 1 because 'slice' is 1-based
    im = np.squeeze(data)  # slice - 1 because 'slice' is 1-based

    if slice == nslices-1:
        # print("Slice Dimensions ", imag.shape)
        print("IMAG raw", data.shape)
        print("IMAG raw", data.dtype)

        # Assuming your 4D image array is named 'image_array'
        imag_type = data.dtype

        # Map NumPy data types to ctypes data types
        ctype = None
        if imag_type == np.uint8:
            ctype = ctypes.c_uint8
        elif imag_type == np.int8:
            ctype = ctypes.c_int8
        elif imag_type == np.uint16:
            ctype = ctypes.c_uint16
        elif imag_type == np.int16:
            ctype = ctypes.c_int16
        elif imag_type == np.uint32:
            ctype = ctypes.c_uint32
        elif imag_type == np.int32:
            ctype = ctypes.c_int32
        elif imag_type == np.uint64:
            ctype = ctypes.c_uint64
        elif imag_type == np.int64:
            ctype = ctypes.c_int64
        elif imag_type == np.float32:
            ctype = ctypes.c_float
        elif imag_type == np.float64:
            ctype = ctypes.c_double

        if ctype is not None:
            print(f"The C raw data type of the imag array elements is {ctype.__name__}")
        else:
            print("Unknown C data type")

        date_path = datetime.today().strftime("%Y-%m-%d")
        # im_path = "/home/sn21/data/t2-stacks/" + date_path + "/SVR-output-brain.nii.gz"
        im_path = "/home/sn21/data/t2-stacks/" + date_path + "-result/grid-reo-SVR-output-brain.nii.gz"

        # Load the NIfTI image
        nib_im = nib.load(im_path)
        header = nib_im.header

        image_r = sitk.ReadImage(im_path)
        image = sitk.GetArrayFromImage(image_r)

        # Print header information
        print("Image Size:", image_r.GetSize())
        print("Image Spacing:", image_r.GetSpacing())
        print("Image Origin:", image_r.GetOrigin())
        print("Image Direction Matrix:")
        print(np.array(image_r.GetDirection()).reshape((3, 3)))

        # Print additional metadata
        metadata_ = image_r.GetMetaDataKeys()
        for key in metadata_:
            print(f"Metadata - {key}: {image_r.GetMetaData(key)}")

        # image = nib_im.get_fdata()

        print("IMAGE SVR reconstruction (loaded)", image.shape)
        print("IMAGE SVR reconstruction (loaded)", image.dtype)

        # Define the matrix size in each dimension (rows, columns, slices for 3D images)
        matrix_size = image.shape  # Replace with your actual matrix size

        # Define the pixel dimensions in each dimension (x, y, z if 3D)
        pixel_dimensions = header['pixdim'][1:4]  # Replace with your actual pixel dimensions

        image = image[..., np.newaxis]

        print("IMAGE SVR reconstruction (new axis)", image.shape)
        print("IMAGE SVR reconstruction (new axis)", image.dtype)

        # Get the shapes of both arrays
        h = image.shape[0]
        w = image.shape[1]
        d = image.shape[2]

        # min_intensity = np.min(imag)

        image = image.astype(imag_type)

        print("IMAGE SVR reconstruction (raw type)", image.shape)
        print("IMAGE SVR reconstruction (raw type)", image.dtype)

        # Assuming your 4D image array is named 'image_array'
        image_type = image.dtype

        # Map NumPy data types to ctypes data types
        ctype = None
        if image_type == np.uint8:
            ctype = ctypes.c_uint8
        elif image_type == np.int8:
            ctype = ctypes.c_int8
        elif image_type == np.uint16:
            ctype = ctypes.c_uint16
        elif image_type == np.int16:
            ctype = ctypes.c_int16
        elif image_type == np.uint32:
            ctype = ctypes.c_uint32
        elif image_type == np.int32:
            ctype = ctypes.c_int32
        elif image_type == np.uint64:
            ctype = ctypes.c_uint64
        elif image_type == np.int64:
            ctype = ctypes.c_int64
        elif image_type == np.float32:
            ctype = ctypes.c_float
        elif image_type == np.float64:
            ctype = ctypes.c_double

        if ctype is not None:
            print(f"The C SVR data type of the image array elements is {ctype.__name__}")
        else:
            print("Unknown C data type")

        # result = imag * image

        print("IMAGE before sending", image.shape)
        print("IMAGE before sending", image.dtype)

        new_matrix_size = image.shape

        im_path = "/home/sn21/data/t2-stacks/" + date_path + "-result/grid-reo-SVR-output-brain.nii.gz"

        directory = '/home/sn21/data/t2-stacks/' + date_path + "-result"
        print(directory)

        # Check if the directory exists and is a directory
        if os.path.isdir(directory):
            # Get a list of files in the directory
            files = os.listdir(directory)

            # Filter for NIfTI files (assuming they have a '.nii' or '.nii.gz' extension)
            nifti_files = [file for file in files if file.endswith('.nii') or file.endswith('.nii.gz')]
            print(nifti_files)

            # Assuming you want to load the first NIfTI file found
            if nifti_files:
                nifti_path = os.path.join(directory, nifti_files[0])
                # nifti_image = nib.load(nifti_path)
                # nifti_image = nifti_image.get_fdata()
                print("NIfTI files found in the directory.")
                # Now, you have the loaded NIfTI image in 'nifti_image'
            else:
                print("No NIfTI files found in the directory.")
        else:
            print(f"The specified directory '{directory}' does not exist or is not a directory.")

        # print("Acquisition", acquisition)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):

        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # date_path = datetime.today().strftime("%Y-%m-%d")
        # # im_path = "/home/sn21/data/t2-stacks/" + date_path + "/SVR-output-brain.nii.gz"
        # im_path = "/home/sn21/data/t2-stacks/" + date_path + "-result/grid-reo-SVR-output-brain.nii.gz"
        #
        # # Load the NIfTI image
        # image = sitk.ReadImage(im_path)
        #
        # # Get spacing along each axis
        # spacing = image.GetSpacing()
        # # Get size along each axis
        # size = image.GetSize()
        # # Calculate the FOV along each axis
        # fov = [spacing[i] * size[i] for i in range(3)]
        # # Define a ctypes array type for 3_float_array_3
        # c_float_array_3 = ctypes.c_float * 3
        # fov = c_float_array_3(*fov)
        # oldHeader.field_of_view = fov  # SVR
        #
        # # Get image position
        # image_position = image.GetOrigin()
        # print("Image position:", image_position)
        # oldHeader.position = image_position  # SVR
        #
        # # Get image orientation
        # image_orientation = image.GetDirection()
        # print("Image orientation:", image_orientation)
        # # Convert the direction matrix to a NumPy array
        # image_orientation = np.array(image_orientation).reshape((3, 3))
        # print("Image orientation:", image_orientation)
        #
        # # Define a ctypes array type for c_float_array_3
        # c_float_array_3 = ctypes.c_float * 3
        #
        # # Create c_float_array_3 instances for a, b, and c
        # read_dir = c_float_array_3(*image_orientation[0, :])
        # phase_dir = c_float_array_3(*image_orientation[1, :])
        # slice_dir = c_float_array_3(*image_orientation[2, :])
        #
        # # Assuming `base_header` is an object with attributes `read_dir`, `phase_dir`, and `slice_dir`
        # oldHeader.read_dir = read_dir
        # oldHeader.phase_dir = phase_dir
        # oldHeader.slice_dir = slice_dir
        #
        # print("Final header:", oldHeader)

        # Unused example, as images are grouped by series before being passed into this function now
        # oldHeader.image_series_index = currentSeries

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'INVERT']
        tmpMeta['WindowCenter']                   = '16384'
        tmpMeta['WindowWidth']                    = '32768'
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
        tmpMeta['Keep_image_geometry']            = 1
        # tmpMeta['ROI_example']                    = create_example_roi(data.shape)

        # Example for setting colormap
        # tmpMeta['LUTFileName']            = 'MicroDeltaHotMetal.pal'

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut


# Create an example ROI <3
def create_example_roi(img_size):
    t = np.linspace(0, 2*np.pi)
    x = 16*np.power(np.sin(t), 3)
    y = -13*np.cos(t) + 5*np.cos(2*t) + 2*np.cos(3*t) + np.cos(4*t)

    # Place ROI in bottom right of image, offset and scaled to 10% of the image size
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))
    x = (x * 0.08*img_size[0]) + 0.82*img_size[0]
    y = (y * 0.10*img_size[1]) + 0.80*img_size[1]

    rgb = (1,0,0)  # Red, green, blue color -- normalized to 1
    thickness  = 1 # Line thickness
    style      = 0 # Line style (0 = solid, 1 = dashed)
    visibility = 1 # Line visibility (0 = false, 1 = true)

    roi = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)
    return roi