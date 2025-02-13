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

        print("im shape is", im.shape)
        im_interleaved = np.zeros((im.shape[0], im.shape[1], nslices))
        print("im interleaved shape", im_interleaved.shape)

        evens = np.arange(0, nslices, 2)
        odds = np.arange(1, nslices, 2)

        if nslices % 2 == 0:
            odd_slices = im[:, :, 0:(nslices // 2)]
            even_slices = im[:, :, (nslices // 2):nslices]
            for j, even_idx in enumerate(evens):
                im_interleaved[:, :, even_idx] = even_slices[:, :, j]
            for j, odd_idx in enumerate(odds):
                im_interleaved[:, :, odd_idx] = odd_slices[:, :, j]
        else:
            even_slices = im[:, :, 0:((nslices + 1) // 2)]
            odd_slices = im[:, :, ((nslices + 1) // 2):nslices]
            for j, even_idx in enumerate(evens):
                im_interleaved[:, :, even_idx] = even_slices[:, :, j]
            for j, odd_idx in enumerate(odds):
                im_interleaved[:, :, odd_idx] = odd_slices[:, :, j]

        im = im_interleaved

        # saving the image as a NIfTI file:
        header = nib.Nifti1Header()
        # timestamp = f"{datetime.today().strftime('%H-%M-%S')}"
        # np.save(debugFolder + "/" + timestamp + "-imgSaved.npy", im)
        header.set_data_shape(data.shape)
        header.set_data_dtype(np.int16)
        print("imheader", imheader)

        dim = imheader.matrix_size
        # header['dim'] = dim

        # pixel dimension = FOV / matrix size
        fov = (ctypes.c_float(imheader.field_of_view[0]),
               ctypes.c_float(imheader.field_of_view[1]),
               ctypes.c_float(imheader.field_of_view[2]))

        pixdim = np.array([imheader.field_of_view[1] / imheader.matrix_size[1],
                           imheader.field_of_view[1] / imheader.matrix_size[1],
                           imheader.field_of_view[2] / imheader.matrix_size[2]])

        header['pixdim'][1:4] = pixdim

        # rotation matrix of the object - read/phase/slice directions
        sform_x = imheader.read_dir
        sform_y = imheader.phase_dir
        sform_z = imheader.slice_dir

        srow_x = (sform_x[0], sform_x[1], sform_x[2])
        srow_y = (sform_y[0], sform_y[1], sform_y[2])
        srow_z = (sform_z[0], sform_z[1], sform_z[2])

        srow_x = (np.round(srow_x, 3))
        srow_y = (np.round(srow_y, 3))
        srow_z = (np.round(srow_z, 3))

        srow_x = (srow_x[0], srow_x[1], srow_x[2])
        srow_y = (srow_y[0], srow_y[1], srow_y[2])
        srow_z = (srow_z[0], srow_z[1], srow_z[2])

        header['pixdim'][0] = sform_x[0]
        header['pixdim'][6] = sform_y[1]
        header['pixdim'][7] = sform_z[2]

        position = imheader.position

        position = (np.int(position[0]), np.int(position[1]), np.int(position[2]))
        position = np.round(position).astype(int)
        position = (position[0], position[1], position[2])

        qoffset_x = position[0]
        qoffset_y = position[1]
        qoffset_z = position[2]

        header['qoffset_x'] = qoffset_x
        header['qoffset_y'] = qoffset_y
        header['qoffset_z'] = qoffset_z

        header['srow_x'] = (srow_x[0], srow_x[1], srow_x[2], position[0])
        header['srow_y'] = (srow_y[0], srow_y[1], srow_y[2], position[1])
        header['srow_z'] = (srow_z[0], srow_z[1], srow_z[2], position[2])

        # srow_x = srow_y
        # srow_y = srow_z
        # srow_z = srow_x

        # header['srow_x'] = (-srow_y[0], -srow_y[1], -srow_y[2], position[0])
        # header['srow_y'] = (srow_z[0], srow_z[1], srow_z[2], position[1])
        # header['srow_z'] = (srow_x[0], srow_x[1], srow_x[2], position[2])

        d_path = datetime.today().strftime("%Y-%m-%d")
        nifti_path = "/home/sn21/data/t2-stacks/" + d_path

        # print(data.shape)
        # rotated_im = np.rot90(im, k=1, axes=(0, 1))
        # noinspection PyTypeChecker
        nifti = nib.Nifti1Image(im, affine=None, header=header)
        nifti.to_filename(nifti_path + "/" + timestamp + "-imgSaved.nii.gz")

    else:
        print("slice", slice)

    # # offline testing
    # im = data
    #
    # # print("im shape is", im.shape)
    # im_interleaved = np.zeros((im.shape[0], im.shape[1], nslices))
    # # print("im interleaved shape", im_interleaved.shape)
    #
    # evens = np.arange(0, nslices, 2)
    # odds = np.arange(1, nslices, 2)
    #
    # if nslices % 2 == 0:
    #     odd_slices = im[:, :, :, :, 0:(nslices // 2)]
    #     even_slices = im[:, :, :, :, (nslices // 2):nslices]
    # else:
    #     even_slices = im[:, :, :, :, 0:((nslices + 1) // 2)]
    #     odd_slices = im[:, :, :, :, ((nslices + 1) // 2):nslices]
    #
    # for j, even_idx in enumerate(evens):
    #     im_interleaved[:, :, even_idx] = np.squeeze(even_slices[:, :, :, :, j])
    #
    # for j, odd_idx in enumerate(odds):
    #     im_interleaved[:, :, odd_idx] = np.squeeze(odd_slices[:, :, :, :, j])
    #
    # im = im_interleaved
    #
    # # saving the image as a NIfTI file:
    # header = nib.Nifti1Header()
    # # timestamp = f"{datetime.today().strftime('%H-%M-%S')}"
    # # np.save(debugFolder + "/" + timestamp + "-imgSaved.npy", im)
    # header.set_data_shape(data.shape)
    # header.set_data_dtype(np.int16)
    # print("imheader", imheader)
    #
    # dim = imheader.matrix_size
    # # header['dim'] = dim
    #
    # # pixel dimension = FOV / matrix size
    # fov = (ctypes.c_float(imheader.field_of_view[0]),
    #        ctypes.c_float(imheader.field_of_view[1]),
    #        ctypes.c_float(imheader.field_of_view[2]))
    #
    # pixdim = np.array([imheader.field_of_view[1] / imheader.matrix_size[1],
    #                    imheader.field_of_view[1] / imheader.matrix_size[1],
    #                    imheader.field_of_view[2] / imheader.matrix_size[2]])
    #
    # header['pixdim'][1:4] = pixdim
    #
    # # rotation matrix of the object - read/phase/slice directions
    # sform_x = imheader.read_dir
    # sform_y = imheader.phase_dir
    # sform_z = imheader.slice_dir
    #
    # srow_x = (sform_x[0], sform_x[1], sform_x[2])
    # srow_y = (sform_y[0], sform_y[1], sform_y[2])
    # srow_z = (sform_z[0], sform_z[1], sform_z[2])
    #
    # srow_x = (np.round(srow_x, 3))
    # srow_y = (np.round(srow_y, 3))
    # srow_z = (np.round(srow_z, 3))
    #
    # srow_x = (srow_x[0], srow_x[1], srow_x[2])
    # srow_y = (srow_y[0], srow_y[1], srow_y[2])
    # srow_z = (srow_z[0], srow_z[1], srow_z[2])
    #
    # header['pixdim'][0] = sform_x[0]
    # header['pixdim'][6] = sform_y[1]
    # header['pixdim'][7] = sform_z[2]
    #
    # # position = imheader.position
    #
    # position = imheader.position
    # position = position[0], position[1], position[2]
    # # # this is for ascending order - create for descending / interleaved slices
    # # slice_thickness = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    # # print("slice_thickness", slice_thickness)
    # # slice_pos = position[1] - ((nslices / 2) - 0.5) * 4.5  # mid-slice position
    # # # pos_z = patient_table_position[2] + position[2]
    # # print("mid slice pos", slice_pos)
    # # print("last position", position[1])
    # # print("pos_z", position[2])
    # # position = position[0], slice_pos, position[2]
    #
    # position = (np.int(position[0]), np.int(position[1]), np.int(position[2]))
    # position = np.round(position).astype(int)
    # position = (position[0], position[1], position[2])
    # #
    # qoffset_x = position[0]
    # qoffset_y = position[1]
    # qoffset_z = position[2]
    #
    # header['qoffset_x'] = qoffset_x
    # header['qoffset_y'] = qoffset_y
    # header['qoffset_z'] = qoffset_z
    #
    # header['srow_x'] = (srow_x[0], srow_x[1], srow_x[2], qoffset_x)
    # header['srow_y'] = (srow_y[0], srow_y[1], srow_y[2], qoffset_y)
    # header['srow_z'] = (srow_z[0], srow_z[1], srow_z[2], qoffset_z)
    #
    # # srow_x = srow_y
    # # srow_y = srow_z
    # # srow_z = srow_x
    #
    # # header['srow_x'] = (-srow_y[0], -srow_y[1], -srow_y[2], position[0])
    # # header['srow_y'] = (srow_z[0], srow_z[1], srow_z[2], position[1])
    # # header['srow_z'] = (srow_x[0], srow_x[1], srow_x[2], position[2])
    #
    # d_path = datetime.today().strftime("%Y-%m-%d")
    # nifti_path = "/home/sn21/data/t2-stacks/" + d_path
    #
    # # print(data.shape)
    # # rotated_im = np.rot90(im, k=1, axes=(0, 1))
    # # noinspection PyTypeChecker
    # nifti = nib.Nifti1Image(im, affine=None, header=header)
    # nifti.to_filename(nifti_path + "/" + timestamp + "-imgSaved.nii.gz")

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

    ###########################################################################
    # THIS IS THE CLASSIFICATION NETWORK

    # start_time = time.time()
    #
    # # note: modify paths to model
    #
    # cl_num_densenet = 2
    # cl_num_unet = 2
    #
    # model_weights_path_densenet = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints"
    #                                "/best_metric_model_densenet.pth")
    # model_weights_path_unet = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/checkpoints"
    #                            "/best_metric_model_unet.pth")
    #
    # ###########################################################################
    #
    # # define and load UNet model
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # segmentation_model = UNet(spatial_dims=3,
    #                           in_channels=1,
    #                           out_channels=cl_num_unet + 1,
    #                           channels=(32, 64, 128, 256, 512),
    #                           strides=(2, 2, 2, 2),
    #                           kernel_size=3,
    #                           up_kernel_size=3,
    #                           num_res_units=1,
    #                           act='PRELU',
    #                           norm='INSTANCE',
    #                           dropout=0.5
    #                           )
    #
    # with torch.no_grad():
    #     segmentation_model.load_state_dict(torch.load(model_weights_path_unet), strict=False)
    #     segmentation_model.to(device)
    #
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Loading the classifier took {elapsed_time} seconds to complete.")
    # directory_path = '/home/sn21/data/t2-stacks/' + date_path
    #
    # for file in os.listdir(directory_path):
    # #     if file.endswith('.nii.gz') or file.endswith('nii'):
    #
    # ###########################################################################
    #
    # start_time = time.time()
    #
    # # # test nii image to create input test image matrix
    # # global_img = nib.load(directory_path + "/" + file)
    #
    # # global_img = im_corr2ab
    # # input_matrix_image_data = global_img.get_fdata()
    # # input_matrix_image_data = im_corr2ab
    # input_matrix_image_data = im
    #
    # ###########################################################################
    #
    # # convert to tensor & apply transforms:
    # # - correct for 4.5 mm slice thickness
    # # - pad to square
    # # - resize to 128x128x128
    # # - scale to [0; 1]
    #
    # input_image = torch.tensor(input_matrix_image_data).unsqueeze(0)
    #
    # zoomer = monai.transforms.Zoom(zoom=(1, 1, 3), keep_size=False)
    # zoomed_image = zoomer(input_image)
    #
    # required_spatial_size = max(zoomed_image.shape)
    # padder = monai.transforms.SpatialPad(spatial_size=required_spatial_size, method="symmetric")
    # padded_image = padder(zoomed_image)
    #
    # # spatial_size = [128, 128, 128]
    # resizer = monai.transforms.Resize(spatial_size=(128, 128, 128))
    # resampled_image = resizer(padded_image)
    #
    # scaler = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    # final_image = scaler(resampled_image)
    #
    # ###########################################################################
    #
    # # run segmentation (multiple times and average to reduce variability) and argmax the labels
    #
    # segmentation_inputs = final_image.unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #
    #     # segmentation_output = sliding_window_inference(segmentation_inputs, (128, 128, 128), 4,
    #     # segmentation_model, overlap=0.8)
    #     segmentation_output1 = segmentation_model(segmentation_inputs)
    #     segmentation_output2 = segmentation_model(segmentation_inputs)
    #     segmentation_output3 = segmentation_model(segmentation_inputs)
    #     segmentation_output4 = segmentation_model(segmentation_inputs)
    #     segmentation_output = (
    #                                   segmentation_output1 + segmentation_output2 + segmentation_output3
    #                                   + segmentation_output4) / 4
    #
    # label_output = torch.argmax(segmentation_output, dim=1).detach().cpu()[0, :, :, :]
    # label_matrix = label_output.cpu().numpy()
    #
    # ###########################################################################
    #
    # # extract brain label and dilate it
    #
    # # extract brain label
    # label_2_mask = label_matrix == 2
    # label_brain = np.where(label_2_mask, label_matrix, 0)
    #
    # # largest connected component
    #
    # labeled_components, num_components = skimage.measure.label(label_brain, connectivity=2,
    #                                                            return_num=True)
    # component_sizes = [np.sum(labeled_components == label) for label in np.unique(labeled_components) if
    #                    label != 0]
    # # largest_component_label = np.argmax(component_sizes) + 1
    # if len(component_sizes) > 0:
    #     largest_component_label = np.argmax(component_sizes) + 1
    # else:
    #     # Handle the case when component_sizes is empty
    #     largest_component_label = 0  # or another appropriate value
    #
    # largest_component_mask = (labeled_components == largest_component_label)
    #
    # # print(num_components)
    #
    # label_brain = largest_component_mask
    # label_brain = label_brain > 0
    #
    # test_zero_brain = np.all(label_brain == 0)
    #
    # if not test_zero_brain:
    #
    #     # dilate
    #     diamond = ndimage.generate_binary_structure(rank=3, connectivity=1)
    #     dilated_label_brain = ndimage.binary_dilation(label_brain, diamond, iterations=5)
    #
    #     ###########################################################################
    #
    #     # transform dilated label to the original padded image & crop padded image
    #
    #     padded_image_matrix = padded_image.cpu().numpy()[0, :, :, :]
    #
    #     # print(padded_image_matrix.shape)
    #     # print(dilated_label_brain.shape)
    #
    #     scale_factors = [
    #         (padded_image_matrix.shape[0] / dilated_label_brain.shape[0]),
    #         (padded_image_matrix.shape[1] / dilated_label_brain.shape[1]),
    #         (padded_image_matrix.shape[2] / dilated_label_brain.shape[2])
    #     ]
    #
    #     dilated_label_brain = ndimage.zoom(dilated_label_brain, zoom=scale_factors, order=0)
    #
    #     # crop padded image
    #     nonzero_indices = np.argwhere(dilated_label_brain == 1)
    #     min_indices = np.min(nonzero_indices, axis=0)
    #     max_indices = np.max(nonzero_indices, axis=0)
    #
    #     cropped_image_matrix = padded_image_matrix[
    #                            min_indices[0]:max_indices[0] + 1, min_indices[1]:max_indices[1] + 1,
    #                            min_indices[2]:max_indices[2] + 1]
    #
    #     ###########################################################################
    #
    #     # pad, resample and scale cropped image
    #
    #     input_cropped_image = torch.tensor(cropped_image_matrix).unsqueeze(0)
    #
    #     required_spatial_size = max(input_cropped_image.shape)
    #     padder = monai.transforms.SpatialPad(spatial_size=required_spatial_size, method="symmetric")
    #     padded_cropped_image = padder(input_cropped_image)
    #
    #     resizer = monai.transforms.Resize(spatial_size=(96, 96, 96))
    #     resampled_cropped_image = resizer(padded_cropped_image)
    #
    #     scaler = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    #     final_cropped_image = scaler(resampled_cropped_image)
    #
    #     ###########################################################################
    #
    #     # define, load and run classifier model
    #
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     classification_model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=cl_num_densenet)
    #
    #     with torch.no_grad():
    #         classification_model.load_state_dict(torch.load(model_weights_path_densenet), strict=False)
    #         classification_model.to(device)
    #
    #     ###########################################################################
    #
    #     # run model and print the class
    #
    #     classifier_intput_image = final_cropped_image.unsqueeze(0).to(device)
    #
    #     with torch.no_grad():
    #         classifier_output_tensor1 = classification_model(classifier_intput_image)
    #         classifier_output_tensor2 = classification_model(classifier_intput_image)
    #         classifier_output_tensor = classifier_output_tensor1 + classifier_output_tensor2 / 2
    #
    #     predicted_probabilities = torch.softmax(classifier_output_tensor, dim=1)
    #     class_out = torch.argmax(predicted_probabilities, dim=1)
    #     predicted_class = class_out.item()
    #
    #     print(" - predicted class : ", predicted_class)
    #
    # else:
    #
    #     #    zero brain mask condition
    #
    #     print(" - predicted class : ", 0)

    ###########################################################################
    return imagesOut


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
