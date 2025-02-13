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

from sympy.physics.units import current

import mrdhelper
import constants
from time import perf_counter
import SimpleITK as sitk
from datetime import datetime
import time
from scipy.optimize import least_squares
import nibabel as nib
from scipy.ndimage import rotate
import subprocess
import pandas as pd
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import aedat
from natsort import natsorted
import shutil

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3, a matrix size of (%s x %s x %s) and echoes %s",
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z,
            metadata.encoding[0].encodingLimits.contrast.maximum+1)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    acqGroup = []
    imgGroup = []

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    dim_x = metadata.encoding[0].encodedSpace.matrixSize.x // 2
    dim_y = metadata.encoding[0].encodedSpace.matrixSize.y

    current_date = datetime.now().strftime("%Y-%m-%d")
    folders = natsorted(os.listdir("/home/sn21/RT_diffusion/" + current_date + "/"))
    prev_file = "/home/sn21/RT_diffusion/" + current_date + "/" + folders[-1] + "/diffusion_reacquisition.dvs"
    input_file = "/home/sn21/RT_diffusion/diffusion.txt"
    output_base_file = "/home/sn21/RT_diffusion/" + current_date + "/" + folders[-1] + "/diffusion_reacquisition_base.dvs"
    nrepetitions = find_nrepetitions(input_file, prev_file, output_base_file)
    print("nrepetitions is !!!!!!!!!!!!!!", nrepetitions)

    print(metadata)
    data_diff = np.zeros((dim_x, dim_y, 1, 1, nslices*nrepetitions))

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
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata, data_diff,nrepetitions)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
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
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

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
            counter = 0
            image = process_image(imgGroup, connection, config, metadata, data_diff, nrepetitions)
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
    phs = [acquisition.idx.phase                for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0], 
                     metadata.encoding[0].encodedSpace.matrixSize.y, 
                     metadata.encoding[0].encodedSpace.matrixSize.x, 
                     max(phs)+1), 
                    group[0].data.dtype)

    rawHead = [None]*(max(phs)+1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:,lin,-acq.data.shape[1]:,phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Remove readout oversampling
    data = fft.ifft(data, axis=2)
    data = np.delete(data, np.arange(int(data.shape[2]*1/4),int(data.shape[2]*3/4)), 2)
    data = fft.fft( data, axis=2)

    logging.debug("Raw data is size after readout oversampling removal %s" % (data.shape,))
    np.save(debugFolder + "/" + "rawNoOS.npy", data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
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
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.y,:]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Measure processing time
    toc = perf_counter()
    strProcessTime = "Total processing time: %.2f ms" % ((toc-tic)*1000.0)
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
        tmpImg = ismrmrd.Image.from_array(data[...,phs], transpose=False)

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to invert image contrast
    imagesOut = process_image(imagesOut, connection, config, metadata,data_diff, nrepetitions)

    return imagesOut


def process_image(images, connection, config, metadata,data_diff, nrepetitions):
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
    #data *= 32767/data.max()
    #data = np.around(data)
    #data = data.astype(np.int16)


    # Invert image contrast
    # data = 32767-data
    data = np.abs(data)
    data = data.astype(np.uint16)
    print("Data shape is!!!!!!!!!!!!!!!!!!!!!!!!", data.shape)

    currentSeries = 0

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

        nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
        slice = oldHeader.slice
        repetition = oldHeader.repetition
        print("slice", slice)
        print("repetition", repetition)

        # WARNING: for pragmatism purposes nrepetitions was made ncontrasts -  NEED TO FIX THIS!!!
        if (slice == nslices - 1 and repetition == nrepetitions-1):
            data_diff = data
            Diff_analysis(data_diff, images, connection, config, metadata, oldHeader, nrepetitions)
            continue

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

def Diff_analysis(data_diff, images, connection, config, metadata, oldHeader, nrepetitions):

    # Functions needed
    def generate_date_folder(T2s_folder_path):

        current_date = datetime.now().strftime("%Y-%m-%d")
        folder_name = f"{current_date}"
        folder_path = os.path.join(T2s_folder_path + folder_name)

        # Create the folder if it doesn't exist
        try:
            os.mkdir(folder_path)
            print(f"Folder '{folder_path}' created.")
        except FileExistsError:
            print(f"Folder '{folder_path}' already exists. Skipping creation.")
        except:
            print(f"An error occured: {e}")

        return folder_path

    def generate_case_folder(day_path):

        num = len(os.listdir(day_path))

        if num == 0:
            case_day_num = 1
        else:
            case_day_num = num + 1

        current_datetime = datetime.now()
        time_part = current_datetime.strftime("%H-%M-%S")
        case_folder = os.path.join(day_path + "/" + time_part)
        print(case_folder)

        try:
            os.mkdir(case_folder)
            print(f"'{case_folder}' created.")
        except FileExistsError:
            print(f"'{case_folder}' already exists. Skipping creation.")
        except:
            print(f"An error occured: {e}")

        return [case_folder, case_day_num]

    def create_dir(directory):

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Directory '{directory}' created successfully.")
            except OSError as e:
                print(f"Error: Failed to create directory '{directory}'. {e}")
        else:
            print(f"Directory '{directory}' already exists.")

    # Create case folder
    Diff_folder_path = "/home/sn21/RT_diffusion/"
    day_path = generate_date_folder(Diff_folder_path)
    case_path, counter = generate_case_folder(day_path)

    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1

    # Create array with dimensions (x,y,nslices,ncontrasts)
    image_diff = np.zeros((data_diff.shape[0], data_diff.shape[1], nslices, nrepetitions))
    evens = np.arange(0,nslices,2)
    odds = np.arange(1,nslices, 2)

    if nslices % 2 == 0:
        for i in range(nrepetitions):
            dyn_array = np.squeeze(data_diff[:, :, :, :, nslices * i:nslices * (i + 1)])
            odd_slices = dyn_array[:, :, 0:(nslices // 2)]
            even_slices = dyn_array[:, :, (nslices // 2):nslices]
            for j, even_idx in enumerate(evens):
                image_diff[:, :, even_idx, i] = even_slices[:, :, j]
            for j, odd_idx in enumerate(odds):
                image_diff[:, :, odd_idx, i] = odd_slices[:, :, j]
    else:
        for i in range(nrepetitions):
            dyn_array = np.squeeze(data_diff[:, :, :, :, nslices* i:nslices * (i + 1)])
            even_slices = dyn_array[:,:,0:((nslices+1)//2)]
            odd_slices = dyn_array[:,:,((nslices+1)//2):nslices]
            for j, even_idx in enumerate(evens):
                image_diff[:, :, even_idx, i] = even_slices[:,:,j]
            for j, odd_idx in enumerate(odds):
                image_diff[:, :, odd_idx, i] = odd_slices[:,:,j]

    # Get orientation
    sform_x = oldHeader.slice_dir
    sform_y = oldHeader.phase_dir
    sform_z = oldHeader.read_dir

    srow_x = (sform_x[0], sform_x[1], sform_x[2])
    srow_y = (sform_y[0], sform_y[1], sform_y[2])
    srow_z = (sform_z[0], sform_z[1], sform_z[2])

    srow_x = (np.round(srow_x, 3))
    srow_y = (np.round(srow_y, 3))
    srow_z = (np.round(srow_z, 3))

    srow_x = (srow_x[0], srow_x[1], srow_x[2])
    srow_y = (srow_y[0], srow_y[1], srow_y[2])
    srow_z = (srow_z[0], srow_z[1], srow_z[2])

    create_dir(case_path + "/nnUNet_images/")
    create_dir(case_path + "/brain_pred/")
    create_dir(case_path + "/brain_PP/")

    for d in range(nrepetitions):
        image_diff_array  = np.transpose(image_diff[:,:,:,d] , (2,0,1))
        im_diff = sitk.GetImageFromArray(image_diff_array)
        im_diff.SetSpacing([3,3,3])
        #srows = srow_x[0], srow_x[1], srow_x[2], srow_y[0], srow_y[1], srow_y[2], srow_z[0], srow_z[1], srow_z[2]
        #im_diff.SetDirection(srows)
        sitk.WriteImage(im_diff, case_path + '/nnUNet_images/dyn' + str(d) + '_0000.nii.gz')


    # Set the DISPLAY and XAUTHORITY environment variables
    os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
    os.environ['XAUTHORITY'] = '/home/sn21/.Xauthority'

    # Fetal Brain
    terminal_command = "gnome-terminal"
    commands_to_execute = ("export nnUNet_raw='/home/jv21/Desktop/nnUNet_folder/nnUNet_raw';"
                           "export nnUNet_preprocessed='/home/jv21/Desktop/nnUNet_folder/nnUNet_preprocessed' ; "
                           "export nnUNet_results='/home/jv21/Desktop/nnUNet_folder/nnUNet_results'; "
                           "conda activate gadgetron ; "
                           "nnUNetv2_predict -i " + case_path + "/nnUNet_images/ -o " + case_path + "/brain_pred/ -d Dataset602_brainsv2 -c 3d_fullres -f 0 1 2 3 4 --disable_tta;"
                                                                                                   "nnUNetv2_apply_postprocessing -i" + case_path + "/brain_pred/ -o" + case_path + "/brain_PP/" + " -pp_pkl_file /home/jv21/Desktop/nnUNet_folder/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/jv21/Desktop/nnUNet_folder/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json")
    full_command = f"{terminal_command} -- bash -c '{commands_to_execute}'"
    os.system(full_command)

    time.sleep(60)

    # Folder for input data (segmentation files) and output data (excel sheet)
    current_date = datetime.now().strftime("%Y-%m-%d")
    folders = natsorted(os.listdir("/home/sn21/RT_diffusion/" + current_date + "/"))
    input_dvs = "/home/sn21/RT_diffusion/" + current_date + "/" + folders[-2] + "/diffusion_reacquisition_base.dvs"
    df_sorted, file_path = process_data(case_path + "/nnUNet_images/", case_path + "/brain_PP/", input_dvs,  case_path + "/analysis_diffusion.xlsx")

    # Compare volume and check for problematic files
    #df_final, problematic_files = compare_local_volumes(df_sorted,)
    df_final = compare_volumes(df_sorted)
    # Calculate movement and L2 Norm
    df_final = calculate_movement(df_final)
    # Calculate Motion Scores
    df_final = calculate_motion_scores(df_final)
    # Add date to excel file
    df_final.to_excel(file_path, index=False)
    # Create L2-Stripe-Representation Plots
    df_final , repeat_dyn= stripe_L2_representation_line(file_path, 'Sheet1', case_path, "/home/sn21/RT_diffusion/DynamicThresholds.xlsx")

    create_priority_list(repeat_dyn, input_dvs, case_path + "/diffusion_reacquisition.dvs")
    shutil.copy(case_path + "/diffusion_reacquisition.dvs", "/home/sn21/freemax-transfer/Jordina/RT_Diffusion/diffusion_reacquisition.dvs")

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

def calculate_volume_fov_com(segmentation_path, segmentation_label=1):
    #Calculate Volume
    segmentation_image = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_image)

    voxel_count = np.sum(segmentation_array == segmentation_label)
    voxel_size = segmentation_image.GetSpacing()
    voxel_volume = np.prod(voxel_size)
    total_volume = voxel_count * voxel_volume

    #Check FOV Boundaries
    z_slices, y_slices, x_slices = segmentation_array.shape

    crossing_positions = []

    # Check upper and lower boundary of z axis
    if np.any(segmentation_array[0, :, :] == segmentation_label):
        crossing_positions.append(('Z', 0))
    if np.any(segmentation_array[-1, :, :] == segmentation_label):
        crossing_positions.append(('Z', z_slices - 1))

    # Check upper and lower boundary of y axis
    if np.any(segmentation_array[:, 0, :] == segmentation_label):
        crossing_positions.append(('Y', 0))
    if np.any(segmentation_array[:, -1, :] == segmentation_label):
        crossing_positions.append(('Y', y_slices - 1))

    # Check upper and lower boundary of x axis
    if np.any(segmentation_array[:, :, 0] == segmentation_label):
        crossing_positions.append(('X', 0))
    if np.any(segmentation_array[:, :, -1] == segmentation_label):
        crossing_positions.append(('X', x_slices - 1))

    within_fov = len(crossing_positions) == 0

    # Calculation of Center of Mass
    image = nib.load(segmentation_path) #loads the image data from the specified file path into a Nifti1Image object
    data = image.get_fdata() #retrieves the image data as a NumPy array from the Nifti1Image object; data = 3D array
    com = center_of_mass(data, labels=segmentation_label) if np.any(data == 1) else (0, 0, 0)

    return total_volume, within_fov, crossing_positions, com


# Process data and create excel file
def process_data(im_folder, seg_folder, input_file, output_file):
    data = []
    im_files = natsorted(os.listdir(im_folder))
    seg_files = natsorted(os.listdir(seg_folder))
    b_values = get_bvalues(input_file)

    for i in range(len(im_files)):
        im_path = im_folder + im_files[i]
        seg_path = seg_folder + seg_files[i]
        volume, within_fov, crossing_positions, com = calculate_volume_fov_com(seg_path)
        black_slices, corrupted_vol = detect_blackslices(im_path, seg_path)
        data.append([i, b_values[i], volume, str(within_fov), com, black_slices, corrupted_vol])

    df = pd.DataFrame(data,
                      columns=['dyn', 'b-value', 'Volume_mm3', 'Within_FOV', 'COM', 'Black_slices', 'Corrupted_vol'])
    df_sorted = df.sort_values(by=['dyn']).reset_index(drop=True)

    df_sorted.to_excel(output_file, index=False)

    print(f'Data saved in {output_file}.')

    return df_sorted, output_file


# Feature to compare volumes (with previous three volumes)
def compare_local_volumes(df, threshold_ratio=0.7):
    avg_volume = round(df['Volume_mm3'].mean(), 2)
    df['avg_Volume'] = avg_volume
    df['Local_Difference'] = np.nan
    problematic_files = []
    counter_badseg = 0

    max_dyn = df['dyn'].max()

    for i in range(len(df)):
        current_volume = df.loc[i, 'Volume_mm3']
        current_dyn = df.loc[i, 'dyn']

        # Calculate volume of the three previous volumes
        if current_dyn == 0:
            prev_volumes = [
                df[df['dyn'] == max_dyn]['Volume_mm3'].values[0],
                df[df['dyn'] == max_dyn - 1]['Volume_mm3'].values[0],
                df[df['dyn'] == max_dyn - 2]['Volume_mm3'].values[0]
            ]
        elif current_dyn == 1:
            prev_volumes = [
                df[df['dyn'] == 0]['Volume_mm3'].values[0],
                df[df['dyn'] == max_dyn]['Volume_mm3'].values[0],
                df[df['dyn'] == max_dyn - 1]['Volume_mm3'].values[0]
            ]
        elif current_dyn == 2:
            prev_volumes = [
                df[df['dyn'] == 0]['Volume_mm3'].values[0],
                df[df['dyn'] == 1]['Volume_mm3'].values[0],
                df[df['dyn'] == max_dyn]['Volume_mm3'].values[0]
            ]
        else:
            prev_volumes = [
                df[df['dyn'] == current_dyn - 3]['Volume_mm3'].values[0],
                df[df['dyn'] == current_dyn - 2]['Volume_mm3'].values[0],
                df[df['dyn'] == current_dyn - 1]['Volume_mm3'].values[0]
            ]

        # Calculate local difference
        local_avg = np.mean(prev_volumes)
        local_difference = current_volume - local_avg
        df.at[i, 'Local_Difference'] = round(local_difference, 0)

        # Check if volume is corrupted
        if current_volume < threshold_ratio * local_avg:
            df.at[i, 'Volume_Change'] = "Volume_Change"
            problematic_files.append((str(df.loc[i, 'dyn']), current_volume / local_avg))
        else:
            df.at[i, 'Volume_Change'] = "No_Change"

        # Calculate Intra-Volume-Score
        volume_ratio = (current_volume / avg_volume)
        if volume_ratio > 1.3 or volume_ratio < 0.7:
            counter_badseg += 1
            df.at[i, 'Seg_Evaluation'] = "bad segmentation"
        else:
            df.at[i, 'Seg_Evaluation'] = "good segmentation"

    df['Intra-Volume-Score'] = counter_badseg / len(df)

    return df, problematic_files


def compare_volumes(df):
    base_volume = df["Volume_mm3"].values[0]
    df['Volume_Change_%'] = np.nan

    for i in range(len(df)):
        current_volume = df.loc[i, 'Volume_mm3']
        current_dyn = df.loc[i, 'dyn']
        volume_difference = base_volume - current_volume
        average_difference = (base_volume + current_volume) / 2
        percentage_drop = (round(np.abs(volume_difference)) / round(average_difference)) * 100
        df.at[i, 'Volume_Change_%'] = round(percentage_drop, 0)

    return df


# Feature to calculate movement of COM
def calculate_movement(df):
    df[['COM_Diff_x', 'COM_Diff_y', 'COM_Diff_z', 'L2_Norm']] = np.nan

    for i in range(1, len(df)):
        com_current = df.loc[i, 'COM']
        com_previous = df.loc[i - 1, 'COM']
        com_diff = [com_current[j] - com_previous[j] for j in range(3)]
        df.at[i, 'COM_Diff_x'] = com_diff[0]
        df.at[i, 'COM_Diff_y'] = com_diff[1]
        df.at[i, 'COM_Diff_z'] = com_diff[2]
        df.at[i, 'L2_Norm'] = round(np.linalg.norm(com_diff), 2)

    return df


def calculate_motion_scores(df):
    # Calculate the average L2 norm for each ID and add it to column
    df['avg_L2'] = round(df['L2_Norm'].mean(), 2)
    df['Inter-Volume-Score'] = df['avg_L2'] / len(df)
    df['Motion_Evaluation'] = ""
    counter_highmotion = 0

    for i in range(len(df)):
        if df.at[i, 'L2_Norm'] > df.at[i, 'avg_L2']:
            df.at[i, 'Motion_Evaluation'] = "high motion"
            counter_highmotion += 1
        else:
            df.at[i, 'Motion_Evaluation'] = "low motion"

    # Step 4: Calculate the High-Motion-Score for all rows
    high_motion_score = counter_highmotion / len(df)
    df['High-Motion-Score'] = high_motion_score

    return df

def detect_blackslices(im_path, seg_path):
    # Set a percentage drop threshold for marking transitions
    drop_percentage_threshold = 0.35

    image_array = sitk.GetArrayFromImage(sitk.ReadImage(im_path))
    segmentation_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    image_array = np.flip(image_array, 0)
    segmentation_array = np.flip(segmentation_array, 0)
    # image_array = image_array.astype(float)  # Convert to float
    image_array[segmentation_array == 0] = np.nan
    slice_means = np.nanmean(image_array, axis=(1, 2))
    volume_mean = np.nanmean(image_array)

    threshold = volume_mean * (1 - drop_percentage_threshold)
    transition_indices = np.where(slice_means < threshold)[0]
    valid_indices = np.where(~np.isnan(slice_means))[0]

    # Edges
    edges = {valid_indices[0], valid_indices[1], valid_indices[-2], valid_indices[-1]}

    # Remove edges from final indices
    final_indices = remove_numbers(transition_indices, edges)

    if final_indices == []:
        corrupted_vol = 0
    else:
        corrupted_vol = 1

    return final_indices, corrupted_vol


def remove_numbers(arr, nums_to_remove):
    # Create a new list excluding the specified numbers
    return [num for num in arr if num not in nums_to_remove]

# Feature to create stripe-representation-plots
def stripe_L2_representation_line(file_path, sheet_name, output_folder, threshold_file):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    repeat_dyn = []
    repeat_dyn.append(0)

    # Ensure we have valid indices and values for plotting
    x_values = df.index.to_numpy()  # Get index as a NumPy array
    y_values = df['L2_Norm'].to_numpy()  # Get L2_Norm as a NumPy array

    # Create stripy type representation
    fig, ax = plt.subplots(figsize=(10, 2))

    # Define colors
    color_corrupted = 'red'  # intra-volume-shift --> change of volume between two slices
    # color_motion = 'orange'  # inter-volume-shift --> COM shift
    color_normal = 'green'
    # color_volume = 'yellow'
    # yellow = 0
    red = 0
    green = 0
    # orange = 0

    thresholds = pd.read_excel(threshold_file)
    threshold_mapping = dict(zip(thresholds['b-value'], thresholds['threshold']))

    # Plotting the slices
    for i, row in df.iterrows():

        b_value = row['b-value']
        threshold = threshold_mapping.get(b_value, None)

        if row['Volume_Change_%'] > threshold:
            color = color_corrupted
            red += 1
            repeat_dyn.append(i)

        elif row['Corrupted_vol'] == 1:
            color = color_corrupted
            red += 1
            repeat_dyn.append(i)
        else:
            color = color_normal
            green += 1
        ax.axvline(i, color=color, linewidth=2)

    ax.plot(x_values, y_values, label='L2-Norm', color='blue')

    df['L2_Norm'] = df['L2_Norm'].fillna(0)
    print(df['L2_Norm'].to_numpy().shape)

    # Customize the plot
    tick_values = list(range(1, len(df), 3))
    tick_values.insert(0, 0)  # Add the value 1 manually after 0
    ax.set_xlim(0, len(df))
    ax.set_ylim(0, 10)
    ax.set_xticks(tick_values)
    ax.set_xlabel('Dynamic')
    ax.set_ylabel('L2-Norm')
    ax.set_title(f'Quality Analysis')
    ax.legend()

    # Adjust layout and save the plot to the specified output folder
    plt.tight_layout()
    plot_filename = os.path.join(output_folder, f'lineplot.png')
    plt.savefig(plot_filename, bbox_inches='tight')

    print("Plots successfully saved to the specified output folder.")
    return df, repeat_dyn


def create_priority_list(repeat_dyn, input_file, output_file):
    vectors_to_keep = repeat_dyn
    directions = len(repeat_dyn)

    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:

        if line.startswith("[directions="):
            modified_lines.append(f"[directions={directions}]\n")
            continue

        if line.startswith("CoordinateSystem") or line.startswith("Normalisation") or line.startswith("comment"):
            modified_lines.append(line)
            continue

        if line.startswith("Vector"):
            vector_index = int(line.split("[")[1].split("]")[0])

            if vector_index in vectors_to_keep:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

def find_nrepetitions(input_file, output_dvs, output_file):

    with open(output_dvs, 'r') as file:
        lines = file.readlines()

    btorepeat = []

    for line in lines:
        if line.startswith("Vector"):
            start = line.find("(") + 1
            end = line.find(")")
            values_str = line[start:end]
            values = [float(v.strip()) for v in values_str.split(",")]
            b_value = values[3]
            if b_value not in btorepeat:
                btorepeat.append(b_value)

    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    counter = 0
    directions = (len(btorepeat)-1)*3 + 1

    for line in lines:

        if line.startswith("[directions="):
            modified_lines.append(f"[directions={directions}]\n")
            continue

        if line.startswith("CoordinateSystem") or line.startswith("Normalisation") or line.startswith("comment"):
            modified_lines.append(line)
            continue

        if line.startswith("Vector"):
            vector_index = int(line.split("[")[1].split("]")[0])
            start = line.find("(") + 1
            end = line.find(")")
            values_str = line[start:end]
            values = [float(v.strip()) for v in values_str.split(",")]
            b_value = values[3]
            print(b_value)

            if b_value in btorepeat:
                line = line.replace(f"Vector[{vector_index}]", f"Vector[{counter}]")
                modified_lines.append(line)
                counter += 1

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

    return (len(btorepeat)-1)*3 + 1

def get_bvalues(input_file):
    b_values = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Vector"):
            vector_index = int(line.split("[")[1].split("]")[0])
            start = line.find("(") + 1
            end = line.find(")")
            values_str = line[start:end]
            values = [float(v.strip()) for v in values_str.split(",")]
            b_value = values[3]
            b_values.append(b_value)

    return b_values
