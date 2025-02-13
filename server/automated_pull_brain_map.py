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


slice_pos = 0


def process_image(images, connection, config, metadata, im):
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images),
                  ismrmrd.get_dtype_from_data_type(images[0].data_type))

    timestamp = f"{datetime.today().strftime('%H-%M-%S')}"

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data for img in images])
    head = [img.getHead() for img in images]
    print("head", head)
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]
    print("meta", meta)

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
    # np.save(debugFolder + "/" + "imgOrig.npy", data)

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

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1

    currentSeries = 0

    slice = imheader.slice

    # online testing

    im[:, :, slice] = np.squeeze(data)  # slice - 1 because 'slice' is 1-based

    if slice == nslices-1:
        # List all files in the directory
        date_path = datetime.today().strftime("%Y-%m-%d")
        directory_path = '/home/sn21/data/t2-stacks/' + date_path
        files = [file for file in os.listdir(directory_path) if file.endswith('.nii.gz')]

        # Count the number of files
        num_files = len(files)

        print(f"There are {num_files} files in the directory.")

        print("Launching docker now...")

        # Set the DISPLAY and XAUTHORITY environment variables
        os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
        os.environ['XAUTHORITY'] = '/home/sn21/.Xauthority'

        # Record the start time
        start_time = time.time()

        # Construct the command for docker
        # date_path = datetime.today().strftime("%Y-%m-%d")

        # command = f'''docker run --rm --mount type=bind,source=/home/sn21/data/t2-stacks,target=/home/data \
        #             fetalsvrtk/svrtk:general_auto_amd sh -c 'bash
        #             /home/auto-proc-svrtk/scripts/auto-brain-055t-reconstruction.sh \
        #             /home/data/{date_path}/dicoms /home/data/{date_path}-result 1 4.5 1.0 1 ;
        #             chmod 1777 -R /home/data/{date_path}-result;
        #             bash /home/auto-proc-svrtk/scripts/auto-body-055t-reconstruction.sh \
        #             /home/data/{date_path}/dicoms /home/data/{date_path}-result-body 1 4.5 1.0 1 ;
        #             chmod 1777 -R /home/data/{date_path}-result-body; ' '''
        #             ' '''

        command = f'''docker run --rm \
                                            --mount type=bind,source=/home/sn21/data/t2-stacks,target=/home/data \
                                            fetalsvrtk/svrtk:general_auto_amd sh -c 'bash /home/auto-proc-svrtk/scripts/auto-brain-055t-reconstruction.sh \
                                            /home/data/{date_path}/dicoms /home/data/{date_path}-result 1 4.5 1.0 1 ; \
                                            chmod 1777 -R /home/data/{date_path}-result; \
                                            /bin/MIRTK/build/lib/tools/pad-3d /home/data/{date_path}-result/reo-SVR-output-brain.nii.gz /home/ref.nii.gz 160 1 ; \
                                            /bin/MIRTK/build/lib/tools/edit-image /home/ref.nii.gz /home/ref.nii.gz -dx 1 -dy 1 -dz 1 ; \
                                            /bin/MIRTK/build/lib/tools/transform-image /home/data/{date_path}-result/reo-SVR-output-brain.nii.gz \
                                            /home/data/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -target /home/ref.nii.gz -interp BSpline ; \
                                            /bin/MIRTK/build/lib/tools/nan /home/data/{date_path}-result/grid-reo-SVR-output-brain.nii.gz 1000000 ; \
                                            /bin/MIRTK/build/lib/tools/convert-image /home/data/{date_path}-result/grid-reo-SVR-output-brain.nii.gz \
                                            /home/data/{date_path}-result/grid-reo-SVR-output-brain.nii.gz -short ; \
                                            chmod 1777 /home/data/{date_path}-result/grid-reo-SVR-output-brain.nii.gz ; \
                                            bash /home/auto-proc-svrtk/scripts/auto-body-055t-reconstruction.sh \
                                            /home/data/{date_path}/dicoms /home/data/{date_path}-result-body 1 4.5 1.0 1 ; \
                                            chmod 1777 -R /home/data/{date_path}-result-body; \
                                            /bin/MIRTK/build/lib/tools/pad-3d /home/data/{date_path}-result-body/reo-DSVR-output-body.nii.gz /home/ref.nii.gz 256 1 ; \
                                            /bin/MIRTK/build/lib/tools/edit-image /home/ref.nii.gz /home/ref.nii.gz -dx 1 -dy 1 -dz 1 ; \
                                            /bin/MIRTK/build/lib/tools/transform-image /home/data/{date_path}-result-body/reo-DSVR-output-body.nii.gz \
                                            /home/data/{date_path}-result-body/grid-reo-DSVR-output-body.nii.gz -target /home/ref.nii.gz -interp BSpline ; \
                                            /bin/MIRTK/build/lib/tools/nan /home/data/{date_path}-result-body/grid-reo-DSVR-output-body.nii.gz 1000000 ; \
                                            /bin/MIRTK/build/lib/tools/convert-image /home/data/{date_path}-result-body/grid-reo-DSVR-output-body.nii.gz \
                                            /home/data/{date_path}-result-body/grid-reo-DSVR-output-body.nii.gz -short ; \
                                            chmod 1777 /home/data/{date_path}-result-body/grid-reo-DSVR-output-body.nii.gz ; \
                                            ' '''

        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
        # os.system(command)

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(f"Execution time: {elapsed_time} seconds")

        # Get today's date
        date = datetime.now().strftime("%Y-%m-%d")

        # Define the file path
        file_path = f"/home/sn21/data/t2-stacks/{date}/execution_time.txt"

        # Save the execution time to a text file
        with open(file_path, "w") as file:
            file.write(f"Execution time: {elapsed_time} seconds")

        print(f"Execution time: {elapsed_time} seconds")
        print(f"Execution time saved to: {file_path}")

    ###########################################################################
    # return imagesOut


# Create an example ROI
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
