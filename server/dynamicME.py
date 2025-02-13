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
import SimpleITK as sitk
from datetime import datetime
import time
from scipy.optimize import least_squares
import nibabel as nib
from scipy.ndimage import rotate
import subprocess

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
    ncontrasts = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    dim_x = metadata.encoding[0].encodedSpace.matrixSize.x // 2
    dim_y = metadata.encoding[0].encodedSpace.matrixSize.y
    data_echoes = np.zeros((dim_x, dim_y, 1, 1, nslices*ncontrasts))

    # Create case folder
    T2s_folder_path = "/home/sn21/RT_T2smapping/"
    day_path = generate_date_folder(T2s_folder_path)
    case_path = generate_case_folder(day_path)

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
                    image = process_image(imgGroup, connection, config, metadata, data_echoes, case_path)
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
            image = process_image(imgGroup, connection, config, metadata, data_echoes,case_path)
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
    imagesOut = process_image(imagesOut, connection, config, metadata,data_echoes, case_path)

    return imagesOut


def process_image(images, connection, config, metadata,data_echoes, case_path):
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

    print("data.......!!!!!!!!!!!", data.shape)
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

        # Accumulate slices for T2* mapping
        ncontrasts = metadata.encoding[0].encodingLimits.contrast.maximum + 1
        nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
        nrepetitions = metadata.encoding[0].encodingLimits.repetition.maximum + 1
        contrast = oldHeader.contrast
        slice = oldHeader.slice
        repetition = oldHeader.repetition

        print("slice is", slice)
        print("contrasts is", contrast)
        print("repetition is", repetition)

        if slice == 0:
            data_echoes[:, :, :, :, contrast] = data[...,iImg]
        else:
            data_echoes[:, :, :, :, ncontrasts * slice + contrast] = data[...,iImg]

        if(slice == nslices - 1 and contrast == ncontrasts -1):
            T2s_mapping(data_echoes, images, connection, config, metadata, oldHeader, repetition, case_path)
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

def T2s_mapping(data_echoes, images, connection, config, metadata, oldHeader, repetition, case_path):

    case_path_dyn = case_path + "/dyn" + str(repetition) +"/"
    create_dir(case_path_dyn)

    # Get number of echoes and slices and TE Values
    ncontrasts = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    nslices = metadata.encoding[0].encodingLimits.slice.maximum + 1
    TE_array = metadata.sequenceParameters.TE

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

    # Create array with dimensions (x,y,nslices,ncontrasts)
    image_echoes = np.zeros((data_echoes.shape[0], data_echoes.shape[1], nslices, ncontrasts))

    for i in range(nslices):
        # Get the indices for the current slice
        start_idx = i * ncontrasts
        end_idx = (i + 1) * ncontrasts
        # Extract the data for the current slice and echoes
        slice_data = np.squeeze(data_echoes[:, :, :, :, start_idx:end_idx])
        # Assign the data to the reshaped array
        image_echoes[:, :, i, :] = slice_data
    print(image_echoes.shape)
    #image_echoes = np.transpose(image_echoes, [1, 0, 2, 3])
    #image_echoes = np.flip(image_echoes,0)
    #image_echoes = np.flip(image_echoes, 2)

    # Save echo 1 data to use for segmentation
    create_dir(case_path_dyn + "/NIFTI/")
    create_dir(case_path_dyn + "/nnUNet_image/")
    im_echo1 = sitk.GetImageFromArray(image_echoes[:, :, :, 0])
    im_echo1.SetSpacing([3.125, 3.125, 3.1])
    srows = srow_x[0], srow_x[1], srow_x[2], srow_y[0], srow_y[1], srow_y[2], srow_z[0], srow_z[1], srow_z[2]
    im_echo1.SetDirection(srows)

    sitk.WriteImage(im_echo1, case_path_dyn + "/NIFTI/Freemax_001_0000.nii.gz")

    # create_dir(case_path + "/nnUNet_image/")

    # Run Fetal brain and placenta prediction with nnUNet

    # Process image with mrtrix3 to change strides
    mrconvert_path = "/home/sn21/miniconda3/bin/mrconvert"

    # Construct command
    commands_to_execute = (
        f"{mrconvert_path} {case_path_dyn}/NIFTI/Freemax_001_0000.nii.gz "
        f"{case_path_dyn}/nnUNet_image/Freemax_001_0000.nii.gz "
        "-strides -1,3,2 -force"
    )

    print(f"Running command: {commands_to_execute}")

    try:
        result = subprocess.run(commands_to_execute, shell=True, capture_output=True, text=True, executable='/bin/bash')
        print("Command executed successfully:")
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError: {fnf_error}")

    # Set the DISPLAY and XAUTHORITY environment variables
    os.environ['DISPLAY'] = ':1'  # Replace with your X11 display, e.g., ':1.0'
    os.environ['XAUTHORITY'] = '/home/sn21/.Xauthority'

    # Placenta
    # For now we leave as the network the CF trained with two echoes (can change this in the future)
    create_dir(case_path_dyn + "/nnUNet_placenta_pred")
    terminal_command = "gnome-terminal"
    commands_to_execute = ("export nnUNet_raw='/home/jv21/Desktop/nnUNet_folder/nnUNet_raw';"
                           "export nnUNet_preprocessed='/home/jv21/Desktop/nnUNet_folder/nnUNet_preprocessed' ; "
                           "export nnUNet_results='/home/jv21/Desktop/nnUNet_folder/nnUNet_results'; "
                           "conda activate gadgetron ; "
                           "nnUNetv2_predict -i " + case_path_dyn + "/nnUNet_image/ -o " + case_path_dyn + "/nnUNet_placenta_pred/ -d 004 -c 3d_fullres -f 1")
    full_command = f"{terminal_command} -- bash -c '{commands_to_execute}'"
    os.system(full_command)

    # Fetal Brain
    create_dir(case_path_dyn + "/nnUNet_brain_pred")
    create_dir(case_path_dyn + "/nnUNet_brain_PP")
    terminal_command = "gnome-terminal"
    commands_to_execute = ("export nnUNet_raw='/home/jv21/Desktop/nnUNet_folder/nnUNet_raw';"
                           "export nnUNet_preprocessed='/home/jv21/Desktop/nnUNet_folder/nnUNet_preprocessed' ; "
                           "export nnUNet_results='/home/jv21/Desktop/nnUNet_folder/nnUNet_results'; "
                           "conda activate gadgetron ; "
                           "nnUNetv2_predict -i " + case_path_dyn + "/nnUNet_image/ -o " + case_path_dyn + "/nnUNet_brain_pred/ -d Dataset602_brainsv2 -c 3d_fullres -f 0 1 2 3 4 --disable_tta;"
                                                                                                   "nnUNetv2_apply_postprocessing -i" + case_path_dyn + "/nnUNet_brain_pred/ -o" + case_path_dyn + "/nnUNet_brain_PP/" + " -pp_pkl_file /home/jv21/Desktop/nnUNet_folder/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/jv21/Desktop/nnUNet_folder/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json")
    full_command = f"{terminal_command} -- bash -c '{commands_to_execute}'"
    os.system(full_command)

    time.sleep(35)

    # Process segmentations with mrtrix3 to change strides back

    mrconvert_path = "/home/sn21/miniconda3/bin/mrconvert"

    #Construct command
    commands_to_execute1 = (
        f"{mrconvert_path} {case_path_dyn}/nnUNet_brain_PP/Freemax_001.nii.gz "
        f"{case_path_dyn}/nnUNet_brain_PP/Freemax_001.nii.gz "
        "-strides -2,-1,-3 -force"
    )

    commands_to_execute2 = (
        f"{mrconvert_path} {case_path_dyn}/nnUNet_placenta_pred/Freemax_001.nii.gz "
        f"{case_path_dyn}/nnUNet_placenta_pred/Freemax_001.nii.gz "
        "-strides -2,-1,-3 -force"
    )

    subprocess.run(commands_to_execute1, shell=True, capture_output=True, text=True, executable='/bin/bash')
    subprocess.run(commands_to_execute2, shell=True, capture_output=True, text=True, executable='/bin/bash')

    #
    # Crop images (all echoes) to fetal brain
    segmentation_brain = sitk.GetArrayFromImage(sitk.ReadImage(case_path_dyn + "/nnUNet_brain_PP/Freemax_001.nii.gz"))
    seg = sitk.GetImageFromArray(segmentation_brain)
    sitk.WriteImage(seg, case_path_dyn + "/NIFTI/image_seg.nii.gz")
    cropped_brain = np.zeros((ncontrasts, nslices * data_echoes.shape[0] * data_echoes.shape[1]))

    # Generate a copy of the array
    image_echoes_b = np.copy(image_echoes)

    # Generate folder to saved cropped/non-cropped images
    #create_dir(case_path + "/NIFTI/")

    for i in range(ncontrasts):
        not_cropped = sitk.GetImageFromArray(np.copy(image_echoes_b[:, :, :, i]))
        not_cropped.CopyInformation(im_echo1)
        image_echoes_b[:, :, :, i][segmentation_brain == 0] = 0
        crop_image = sitk.GetImageFromArray(image_echoes_b[:, :, :, i])
        seg = sitk.GetImageFromArray(segmentation_brain)
        sitk.WriteImage(not_cropped, case_path_dyn + "/NIFTI/image_echo_nc_brain_" + str(i) + ".nii.gz")
        sitk.WriteImage(crop_image, case_path_dyn + "/NIFTI/image_echo_brain_" + str(i) + ".nii.gz")
        flat_array = np.ndarray.flatten(image_echoes_b[:, :, :, i])
        cropped_brain[i, :] = flat_array

    # Crop images (all echoes) to placenta
    segmentation_placenta = sitk.GetArrayFromImage(sitk.ReadImage(case_path_dyn + "/nnUNet_placenta_pred/Freemax_001.nii.gz"))
    cropped_placenta = np.zeros((ncontrasts, nslices * data_echoes.shape[0] * data_echoes.shape[1]))

    # Generate a copy of the array
    image_echoes_p = np.copy(image_echoes)

    # Generate folder to saved cropped/non-cropped images
    #create_dir(case_path + "/NIFTI/")

    for i in range(ncontrasts):
        not_cropped = sitk.GetImageFromArray(np.copy(image_echoes_p[:, :, :, i]))
        image_echoes_p[:, :, :, i][segmentation_placenta == 0] = 0
        crop_image = sitk.GetImageFromArray(image_echoes_p[:, :, :, i])
        sitk.WriteImage(not_cropped, case_path_dyn + "/NIFTI/image_echo_nc_placenta_" + str(i) + ".nii.gz")
        sitk.WriteImage(crop_image, case_path_dyn + "/NIFTI/image_echo_placenta_" + str(i) + ".nii.gz")
        flat_array = np.ndarray.flatten(image_echoes_p[:, :, :, i])
        cropped_placenta[i, :] = flat_array

    # Find nonzero voxel idx for brain
    nz_idx_brain = []

    for i in range(cropped_brain.shape[1]):
        sol = (cropped_brain[:, i] != 0).all()
        if sol == True:
            nz_idx_brain.append(i)
        else:
            pass

    # Find nonzero voxel idx for placenta
    nz_idx_placenta = []

    for i in range(cropped_placenta.shape[1]):
        sol = (cropped_placenta[:, i] != 0).all()
        if sol == True:
            nz_idx_placenta.append(i)
        else:
            pass

    # Calculate T2* maps for the ROIS

    T2s_map_brain = np.zeros((2, nslices * data_echoes.shape[0] * data_echoes.shape[1]))
    T2s_map_placenta = np.zeros((2, nslices * data_echoes.shape[0] * data_echoes.shape[1]))

    logging.info("Calculation of T2* map fetal brain starting...")

    for i in range(len(nz_idx_brain)):
        pix_array_brain = cropped_brain[:, nz_idx_brain[i]]
        param_init_brain = np.squeeze([pix_array_brain[0], np.average(TE_array)])
        result_brain = least_squares(t2fit, param_init_brain, args=(pix_array_brain, TE_array),
                                     bounds=([0, 0], [10000, 1000]))
        T2s_map_brain[0, nz_idx_brain[i]] = result_brain.x[0]
        T2s_map_brain[1, nz_idx_brain[i]] = result_brain.x[1]

    T2s_val_brain = np.reshape(T2s_map_brain[1, :], [data_echoes.shape[0], data_echoes.shape[1], nslices])
    image_t2s_brain = sitk.GetImageFromArray(T2s_val_brain)
    image_t2s_brain.CopyInformation(im_echo1)
    sitk.WriteImage(image_t2s_brain, case_path_dyn + "/fbrain_t2smap.nii.gz")

    t2s_brain = T2s_map_brain[1, :]
    t2s_brain_nz = t2s_brain[t2s_brain != 0]
    t2s_brain_limit = t2s_brain_nz[t2s_brain_nz < 900]
    print("Mean T2* for the fetal brain is", np.mean(t2s_brain_limit))

    logging.info("Calculation of T2* map placenta starting...")

    for i in range(len(nz_idx_placenta)):
        pix_array_placenta = cropped_placenta[:, nz_idx_placenta[i]]
        param_init_placenta = np.squeeze([pix_array_placenta[0], np.average(TE_array)])
        result_placenta = least_squares(t2fit, param_init_placenta, args=(pix_array_placenta, TE_array),
                                        bounds=([0, 0], [10000, 1000]))
        T2s_map_placenta[0, nz_idx_placenta[i]] = result_placenta.x[0]
        T2s_map_placenta[1, nz_idx_placenta[i]] = result_placenta.x[1]

    T2s_val_placenta = np.reshape(T2s_map_placenta[1, :], [data_echoes.shape[0], data_echoes.shape[1], nslices])
    image_t2s_placenta = sitk.GetImageFromArray(T2s_val_placenta)
    image_t2s_placenta.CopyInformation(im_echo1)
    sitk.WriteImage(image_t2s_placenta, case_path_dyn + "/placenta_t2smap.nii.gz")

    t2s_placenta = T2s_map_placenta[1, :]
    t2s_placenta_nz = t2s_placenta[t2s_placenta != 0]
    t2s_placenta_limit = t2s_placenta_nz[t2s_placenta_nz < 500]
    print("Mean T2* for the placenta is", np.mean(t2s_placenta_limit))

    # Save mean T2* and centiles in a document
    # GA = 36
    # output_path = case_path + "Centile_report.docx"
    # generate_T2s_report(t2s_brain_limit, t2s_placenta_limit, GA, output_path)


def generate_T2s_report(t2s_brain, t2s_placenta, GA, output_path):
    # Excel sheet
    df_brain = pd.read_excel("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/centiles_055.xlsx",
                       sheet_name="brain_t2s_mean_055")
    df_placenta = pd.read_excel("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/centiles_055.xlsx",
                       sheet_name="placenta_t2s_mean_055")

    centiles_range_brain = df_brain[0]
    centiles_range_placenta = df_placenta[0]
    t2s_centiles_brain = df_brain[round(GA)]
    t2s_centiles_placenta = df_placenta[round(GA)]

    diff_brain = abs(t2s_centiles_brain - t2s_brain)
    diff_placenta = abs(t2s_centiles_placenta -t2splacenta)

    diff_min_brain = np.argmin(diff_brain)
    diff_min_placenta = np.argmin(diff_placenta)

    doc = Document()
    title = doc.add_paragraph("T2* report")
    title.runs[0].bold = True
    title.runs[0].underline = True
    title.runs[0].font.size = docx.shared.Pt(15)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    doc.add_paragraph("Fetal Brain mean T2* is " + str(t2s_brain))
    doc.add_paragraph(
        "Fetal Brain mean T2* is in the " + str(centiles_range_brain[diff_min_brain]) + " th centile for " + str(GA) + " weeks")
    doc.add_paragraph("Placenta mean T2* is " + str(t2s_placenta))
    doc.add_paragraph(
        "Placenta mean T2* is in the " + str(centiles_range_placenta[diff_min_placenta]) + " th centile for " + str(GA) + " weeks")
    doc.save(output_path)

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

    return case_folder

def create_dir(directory):

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        except OSError as e:
            print(f"Error: Failed to create directory '{directory}'. {e}")
    else:
        print(f"Directory '{directory}' already exists.")

def t2fit(X, data, TEs):
    TEs = np.array(TEs, dtype=float)
    X = np.array(X, dtype=float)
    S = X[0] * ((np.exp(-(TEs / X[1]))))
    return data - S