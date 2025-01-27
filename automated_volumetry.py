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
# import gadgetron
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
# debugFolder = "/tmp/share/debug"
debugFolder = "/home/data/volumetry/"


def centile_graphs(roi):
    a = 0
    b = 0
    c = 0
    a5 = 0
    b5 = 0
    c5 = 0
    title = ""

    roi_cmp = "fetus"
    if roi == roi_cmp:
        a = 0.00
        b = 206.99
        c = -4785.8
        a5 = 0.00
        b5 = -8.597625
        c5 = 614.8
        title = "Fetal volume"

    roi_cmp = "placenta"
    if roi == roi_cmp:
        a = 0.00
        b = -10.315
        c = 1246.4
        a5 = 0.00
        b5 = 0.157625
        c5 = 156.25
        title = "Placenta volume"

    roi_cmp = "amniotic"
    if roi == roi_cmp:
        a = 0.00
        b = -17.068
        c = 1201
        a5 = 0.00
        b5 = -0.807875
        c5 = 251.25
        title = "Amniotic fluid volume"

    roi_cmp = "efw"
    if roi == roi_cmp:
        a = 0.0
        b = 213.41
        c = -4934
        a5 = 0.00
        b5 = -8.878625
        c5 = 634.4
        title = "Estimated fetal weight (Baker et al.)"

    x = np.linspace(35, 42, 100)
    y = a * x * x + b * x + c
    y_s = a5 * x * x + b5 * x + c5

    y5 = y - 1.645 * y_s
    y95 = y + 1.645 * y_s

    return x, y, y5, y95, title


def plot_centiles4(id, scan_date, ga, fetus, placenta, amniotic, efw, volumetry_path, timestamp):

    # date_path = datetime.today().strftime("%Y-%m-%d")
    # timestamp = f"{datetime.today().strftime('%H-%M-%S')}"
    # volumetry_path = ("/home/sn21/miniconda3/envs/gadgetron/share/gadgetron/python/results/volumetry/"
    #                   + date_path)

    fig = make_subplots(rows=2,
                        cols=2,
                        vertical_spacing=0.2,
                        horizontal_spacing=0.2,
                        subplot_titles=("Fetal volume", "Estimated fetal weight (Baker et al.)", "Placenta volume",
                                        "Amniotic fluid volume"))

    m_size = 10

    s_r = 1
    s_c = 1
    roi = "fetus"
    vol = fetus
    x, y, y5, y95, title = centile_graphs(roi)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y95, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=[ga], y=[vol], mode='markers', marker_color='red', marker_size=m_size, opacity=0.8,
                             marker_symbol='x'), row=s_r, col=s_c)
    fig.update_xaxes(title_text="GA [weeks]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)
    fig.update_yaxes(title_text="Volume [cc]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)

    y_total = y
    y_total_5 = y5
    y_total_95 = y95
    vol_total = vol

    s_r = 2
    s_c = 1
    roi = "placenta"
    vol = placenta
    x, y, y5, y95, title = centile_graphs(roi)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y95, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=[ga], y=[vol], mode='markers', marker_color='red', marker_size=m_size, opacity=0.8,
                             marker_symbol='x'), row=s_r, col=s_c)
    fig.update_xaxes(title_text="GA [weeks]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)
    fig.update_yaxes(title_text="Volume [cc]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)

    y_total = y_total + y
    y_total_5 = y_total_5 + y5
    y_total_95 = y_total_95 + y95
    vol_total = vol_total + vol

    s_r = 2
    s_c = 2
    roi = "amniotic"
    vol = amniotic
    x, y, y5, y95, title = centile_graphs(roi)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y95, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=[ga], y=[vol], mode='markers', marker_color='red', marker_size=m_size, opacity=0.8,
                             marker_symbol='x'), row=s_r, col=s_c)
    fig.update_xaxes(title_text="GA [weeks]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)
    fig.update_yaxes(title_text="Volume [cc]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)

    y_total = y_total + y
    y_total_5 = y_total_5 + y5
    y_total_95 = y_total_95 + y95
    vol_total = vol_total + vol

    s_r = 1
    s_c = 2
    roi = "efw"
    vol = efw
    x, y, y5, y95, title = centile_graphs(roi)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y5, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=x, y=y95, mode='lines', line_color='grey'), row=s_r, col=s_c)
    fig.add_trace(go.Scatter(x=[ga], y=[vol], mode='markers', marker_color='red', marker_size=m_size, opacity=0.8,
                             marker_symbol='x'), row=s_r, col=s_c)
    fig.update_xaxes(title_text="GA [weeks]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)
    fig.update_yaxes(title_text="Weight [g]", gridcolor='lightgrey', nticks=10, row=s_r, col=s_c)

    # title = "Uterus volumetry: " + id + " / " + str(ga) + " weeks / " + scan_date

    fig.update_layout(height=700,
                      width=900,
                      showlegend=False,
                      plot_bgcolor='white',
                      # title_text=title,
                      # title_font_family="Arial Black",
                      )

    fig.show()

    fig.write_image(volumetry_path + "/" + timestamp + "-graphs.png")


def subject_percentile(roi, ga, y_subject):
    a = 0
    b = 0
    c = 0
    a5 = 0
    b5 = 0
    c5 = 0
    title = ""

    a = 0
    b = 0
    c = 0
    a5 = 0
    b5 = 0
    c5 = 0
    title = ""

    roi_cmp = "fetus"
    if roi == roi_cmp:
        a = 0.00
        b = 206.99
        c = -4785.8
        a5 = 0.00
        b5 = -8.597625
        c5 = 614.8
        title = "Fetal volume"

    roi_cmp = "placenta"
    if roi == roi_cmp:
        a = 0.00
        b = -10.315
        c = 1246.4
        a5 = 0.00
        b5 = 0.157625
        c5 = 156.25
        title = "Placenta volume"

    roi_cmp = "amniotic"
    if roi == roi_cmp:
        a = 0.00
        b = -17.068
        c = 1201
        a5 = 0.00
        b5 = -0.807875
        c5 = 251.25
        title = "Amniotic fluid volume"

    roi_cmp = "efw"
    if roi == roi_cmp:
        a = 0.0
        b = 213.41
        c = -4934
        a5 = 0.00
        b5 = -8.878625
        c5 = 634.4
        title = "Estimated fetal weight (Baker et al.)"

    x = ga
    y_ga = a * x * x + b * x + c

    y5 = y_ga - 1.645 * (a5 * x * x + b5 * x + c5)
    y95 = y_ga + 1.645 * (a5 * x * x + b5 * x + c5)

    sd_ga = np.polyval([a5, b5, c5], ga)

    z_score = (y_subject - y_ga) / sd_ga

    percentile = norm.cdf(z_score) * 100

    return percentile, z_score


def extract_label(lab_nii_raw, l1, l2=1000, l3=1000, l4=1000, l5=1000, l6=1000):
    x_dim, y_dim, z_dim = lab_nii_raw.shape
    lab_nii_raw_out = lab_nii_raw

    for x in range(1, x_dim, 1):
        for y in range(1, y_dim, 1):
            for z in range(1, z_dim, 1):
                if lab_nii_raw[x, y, z] == l1 or lab_nii_raw[x, y, z] == l2 or lab_nii_raw[x, y, z] == l3 or \
                        lab_nii_raw[x, y, z] == l4 or lab_nii_raw[x, y, z] == l5:
                    lab_nii_raw_out[x, y, z] = 1
                else:
                    lab_nii_raw_out[x, y, z] = 0

    return lab_nii_raw_out


def resample_to_isotropic(input_filepath, output_filepath, new_resolution=(1.5, 1.5, 1.5)):
    # Load the image
    img = nib.load(input_filepath)
    data = img.get_fdata()
    affine = img.affine

    # Get the original resolution
    original_resolution = img.header.get_zooms()

    # Calculate the resampling factor
    resampling_factors = [orig / new for orig, new in zip(original_resolution, new_resolution)]

    # Calculate the new shape
    new_shape = np.ceil(np.array(data.shape) * resampling_factors).astype(int)

    # Resample the data
    resampled_data = zoom(data, resampling_factors, order=0)

    # Adjust the affine transformation
    new_affine = affine.copy()
    scale_affine = np.diag(resampling_factors + [1])
    new_affine[:3, :3] = np.dot(affine[:3, :3], scale_affine[:3, :3])

    # Create a new NIfTI image
    resampled_img = nib.Nifti1Image(resampled_data, new_affine)

    # Save the resampled image
    nib.save(resampled_img, output_filepath)

    return resampled_data


def compute_label_volume(lab_nii, lab_nii_raw, l_num):
    x_dim, y_dim, z_dim = lab_nii.shape
    dx, dy, dz = lab_nii.header.get_zooms()
    n = 0
    for x in range(1, x_dim, 1):
        for y in range(1, y_dim, 1):
            for z in range(1, z_dim, 1):
                if lab_nii_raw[x, y, z] == l_num:
                    n = n + 1
    vol = n * dx * dy * dz / 1000
    return vol


def compute_btfe_label_volume(lab_nii, lab_nii_raw):
    fetus = compute_label_volume(lab_nii, lab_nii_raw, 1) + compute_label_volume(lab_nii, lab_nii_raw, 5)
    placenta = compute_label_volume(lab_nii, lab_nii_raw, 2)
    amniotic = compute_label_volume(lab_nii, lab_nii_raw, 4)
    cord = compute_label_volume(lab_nii, lab_nii_raw, 3)

    rr = 4
    fetus = round(fetus, rr)
    placenta = round(placenta, rr)
    amniotic = round(amniotic, rr)
    cord = round(cord, rr)

    return fetus, placenta, amniotic, cord


def compute_fetal_weight(fetus):
    fetal_body_volume = fetus * 0.001
    baker = 1.031 * fetal_body_volume + 0.12
    kacem = 0.989 * fetal_body_volume + 0.147

    return baker, kacem


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
    # np.save(debugFolder + "/" + "raw.npy", data)

    # Remove readout oversampling
    data = fft.ifft(data, axis=2)
    data = np.delete(data, np.arange(int(data.shape[2] * 1 / 4), int(data.shape[2] * 3 / 4)), 2)
    data = fft.fft(data, axis=2)

    logging.debug("Raw data is size after readout oversampling removal %s" % (data.shape,))
    # np.save(debugFolder + "/" + "rawNoOS.npy", data)

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
    # np.save(debugFolder + "/" + "img.npy", data)

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
    # np.save(debugFolder + "/" + "imgCrop.npy", data)

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
    # np.save(debugFolder + "/" + "imgInverted.npy", data)

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

    # Define the path where the results will be saved
    volumetry_path = debugFolder + date_path

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

    # Run prediction with nnUNet
    os.environ['DISPLAY'] = ':0'  # Replace with your X11 display, e.g., ':1.0'
    os.environ["XAUTHORITY"] = '/opt/code/automated-fetal-mri/.Xauthority'
    # Record the start time
    start_time = time.time()

    # timestamp = "18-14-01"

    # Define the terminal command for prediction
    terminal_command = (("export nnUNet_raw='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_raw'; export "
                         "nnUNet_preprocessed='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_preprocessed'; "
                         "export nnUNet_results='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_results'; "
                         "conda activate gadgetron; nnUNetv2_predict -i ") + volumetry_path + "/"
                        + timestamp + "-nnUNet_seg-volumetry/ -o " + volumetry_path + "/" + timestamp
                        + "-nnUNet_pred-volumetry/ -d 084 -c 3d_fullres -f 1")

    # Run the terminal command
    subprocess.run(terminal_command, shell=True)  # Sara!

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for fetal body localisation: {elapsed_time} seconds")

    # Load the segmentation and image volumes

    segmentation_filename = os.path.join(volumetry_path, timestamp + "-nnUNet_pred-volumetry",
                                         "Volumetry_001.nii.gz")

    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string
    date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the file name with the formatted date and time
    text_file_1 = (debugFolder + date_path + "/" + timestamp + "-volumetry.txt")
    # text_file = "/home/sn21/freemax-transfer/Sara/landmarks-interface-autoplan/" + timestamp + "-volumetry.txt"

    file_ga = "/home/sn21/freemax-transfer/Sara/volumetry-ga-interface/ga.txt"

    # # Create and write to the text file
    # with open(text_file, "w") as file:
    #     file.write("This is a text file created on " + date_time_string)
    #
    # with open(text_file_1, "w") as file:
    #     file.write("This is a text file created on " + date_time_string)
    #
    # print(f"Text file '{text_file}' has been created.")

    # INSERT HERE REPORT CODE

    # load .nii images in NiBabel format

    img_name = volumetry_path + "/" + timestamp + "-nnUNet_seg-volumetry/Volumetry_001_0000.nii.gz"

    lab_name = segmentation_filename

    img_nii = nib.load(img_name)
    lab_nii = nib.load(lab_name)

    # extract matrices only
    img_raw = img_nii.get_fdata()
    lab_raw = lab_nii.get_fdata()
    #
    ga = None

    with open(file_ga, 'r') as file:
        content = file.read()

        # Search for the line containing 'ga' in the format 'weeks+days'
        match = re.search(r'ga\s*=\s*(\d+)\+(\d+)', content)
        print("MATCH", match)
        if match:
            weeks = int(match.group(1))  # Extract the weeks part
            days = int(match.group(2))  # Extract the days part
            ga = weeks + days / 7.0  # Convert to total weeks as a float

    # Print the extracted GA value
    if ga is not None:
        print(f"Extracted GA value: {ga:.2f} weeks")
    else:
        print("GA value not found.")

    id = "FetalScan"
    scan_date = date_path
    #
    # check dimensions and voxel spacing
    print("Image: shape =", img_nii.shape, ", voxel spacing =", img_nii.header.get_zooms(), "mm")
    print("Label: shape =", lab_nii.shape, ", voxel spacing =", lab_nii.header.get_zooms(), "mm")
    print()

    fetus, placenta, amniotic, cord = compute_btfe_label_volume(lab_nii, lab_raw)

    efw = compute_fetal_weight(fetus)
    baker = efw[0]
    baker = np.round(baker, 4)
    kacem = efw[1]
    kacem = np.round(kacem, 4)

    print("EFW (Baker)", baker, "EFW (Kacem)", kacem)

    f = plt.figure(figsize=(12, 4))

    min_val_for_display = 0
    max_val_for_display = img_raw.max() * 0.8

    plot_roi(img_nii,
             bg_img=img_nii,
             #  title="MRI image with label overlay",
             dim=0,
             cmap='gray',
             vmin=0,
             figure=f,
             display_mode='ortho',
             #  vmax=1,
             black_bg=True)

    plt.savefig(volumetry_path + "/" + timestamp + '-out-rad-grey.png')

    f = plt.figure(figsize=(12, 4))

    plot_roi(lab_nii,  # main image: label
             bg_img=img_nii,  # background image: MRA
             alpha=0.5,  # label opacity
             #  title="MRA image with label overlay",
             dim=-0.5,
             cmap='jet',
             resampling_interpolation='nearest',
             vmin=0,
             figure=f,
             display_mode='ortho',
             #  vmax=1,
             #  colorbar=True,
             black_bg=True)

    plt.savefig(volumetry_path + "/" + timestamp + '-out-rad-with-lab.png')

    f = plt.figure(figsize=(20, 4))

    plot_roi(lab_nii,  # main image: label
             bg_img=img_nii,  # background image: MRA
             alpha=0.5,  # label opacity
             #  title="MRA image with label overlay",
             #  dim=-0.5,
             cmap='jet',
             resampling_interpolation='nearest',
             vmin=0,
             #  axes=(0,0, 8, 4),
             figure=f,
             annotate=False,
             display_mode='y',
             #  cut_coords=2,
             #  colorbar=True,
             black_bg=True)

    plt.savefig(volumetry_path + "/" + timestamp + '-out-rad-with-lab-coronal.png')

    f = plt.figure(figsize=(12, 4))

    plot_roi(img_nii,
             bg_img=img_nii,
             vmax=max_val_for_display,
             #  title="MRI image with label overlay",
             dim=0,
             cmap='gray',
             vmin=0,
             figure=f,
             display_mode='y',
             cut_coords=7,
             black_bg=True)

    res_lab_raw = resample_to_isotropic(lab_name, volumetry_path + "/" + timestamp + "-res_lab.nii.gz")

    lab_fetus = extract_label(res_lab_raw, 1, 5)

    verts_fe, faces_fe, normals_fe, values_fe = measure.marching_cubes(lab_fetus, 0)

    lighting = dict(ambient=0.5, diffuse=0.5, roughness=0.5, specular=0.6, fresnel=0.8)

    x, y, z = verts_fe.T
    I, J, K = faces_fe.T
    fetus_mesh = go.Mesh3d(x=x, y=y, z=z,
                           intensity=values_fe,
                           i=I, j=J, k=K,
                           name='Fetus',
                           lighting=lighting,
                           showscale=False,
                           opacity=1.0,
                           colorscale='pinkyl'
                           )

    camera = dict(eye=dict(x=1.0, y=1.0, z=1.0))

    # PlotLy figure layout
    layout = go.Layout(
        width=900,
        height=300,
        margin=dict(t=1, l=1, b=1),
        # title=("Fetus in 3D"),
    )

    fig = go.Figure(data=[fetus_mesh], layout=layout)

    # update figure layout
    fig.update_layout(scene_xaxis_visible=False,
                      scene_yaxis_visible=False,
                      scene_zaxis_visible=False,

                      scene_camera=camera,
                      )

    # display
    fig.show()

    # Save the combined figure as a PNG file
    pio.write_image(fig, volumetry_path + "/" + timestamp + '-fetus_3D.png')

    efw = fetus * 1.031 + 0.12

    plot_centiles4(id, scan_date, ga, fetus, placenta, amniotic, efw, volumetry_path, timestamp)

    # print("fetal_volume", subject_percentile("fetus", ga, fetus))
    # print("placenta_volume", subject_percentile("placenta", ga, placenta))
    # print("amniotic_volume", subject_percentile("amniotic", ga, amniotic))
    # print("efw", subject_percentile("efw", ga, efw))

    percentile_fetus, z_score_fetus = subject_percentile("fetus", ga, fetus)
    percentile_placenta, z_score_placenta = subject_percentile("placenta", ga, placenta)
    percentile_amniotic, z_score_amniotic = subject_percentile("amniotic", ga, amniotic)
    percentile_efw, z_score_efw = subject_percentile("efw", ga, efw)

    img = io.imread(volumetry_path + "/" + timestamp + '-out-rad-grey.png')
    # Check and convert boolean array if necessary
    if img.dtype == bool:
        img = img.astype(np.uint8)  # Convert boolean to uint8 if needed
    figm1 = px.imshow(img)
    # figm1 = io.imread(volumetry_path + "/" + timestamp + '-out-rad-grey.png')

    img = io.imread(volumetry_path + "/" + timestamp + '-out-rad-with-lab.png')
    # Check and convert boolean array if necessary
    if img.dtype == bool:
        img = img.astype(np.uint8)  # Convert boolean to uint8 if needed
    figm2 = px.imshow(img)
    # figm2 = io.imread(volumetry_path + "/" + timestamp + '-out-rad-with-lab.png')

    img = io.imread(volumetry_path + "/" + timestamp + '-fetus_3D.png')
    # Check and convert boolean array if necessary
    if img.dtype == bool:
        img = img.astype(np.uint8)  # Convert boolean to uint8 if needed
    figm3 = px.imshow(img)
    # figm3 = io.imread(volumetry_path + "/" + timestamp + '-fetus_3D.png')

    fig = make_subplots(
        rows=4, cols=1, horizontal_spacing=0.01,
        vertical_spacing=0.001,
        specs=[[{"type": "image"}],
               [{"type": "image"}],
               [{"type": "image"}],
               [{"type": "table"}]
               ])

    fig.add_trace(figm1.data[0], row=1, col=1)
    fig.add_trace(figm2.data[0], row=2, col=1)

    fig.add_trace(figm3.data[0], row=3, col=1)

    fig.add_trace(
        go.Table(header=dict(font_size=14, values=['Segmentation ROI', 'Measurement', 'Percentile', 'Z-score']),
                 cells=dict(fill_color='white', font_size=14, line_color='lightgray',
                            values=[["Fetus", "Placenta", "Amniotic fluid", "EFW (Baker et al.)"],
                                    [(round(fetus, 2), "cc"), (round(placenta, 2), "cc"), (round(amniotic, 2), "cc"),
                                     ((round(fetus * 1.031 + 0.12, 2)), "g")],
                                    [round(percentile_fetus, 3), round(percentile_placenta, 3),
                                     round(percentile_amniotic, 3), round(percentile_efw, 3)],
                                    [round(z_score_fetus, 3), round(z_score_placenta, 3), round(z_score_amniotic, 3),
                                     round(z_score_efw, 3)]])),
        row=4, col=1
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    title = "Internal uterus volumetry: " + id + " / " + str(ga) + " weeks / " + scan_date

    fig.update_layout(
        height=1414,
        width=1000,
        showlegend=False,
        plot_bgcolor='white',
        title_text=title,
        # title_font_family="Arial Black",
    )

    fig.show()

    fig.write_image(volumetry_path + "/" + timestamp + "-report1.png")

    img = io.imread(volumetry_path + "/" + timestamp + '-graphs.png')
    # Check and convert boolean array if necessary
    if img.dtype == bool:
        img = img.astype(np.uint8)  # Convert boolean to uint8 if needed
    figm5 = px.imshow(img)
    # figm5 = io.imread(volumetry_path + "/" + timestamp + '-graphs.png')

    fig2 = make_subplots(
        rows=1, cols=1, horizontal_spacing=0.01,
        vertical_spacing=0.001,
        specs=[[{"type": "image"}]
               ])

    fig2.add_trace(figm5.data[0], row=1, col=1)

    title = "Internal uterus volumetry: " + id + " / " + str(ga) + " weeks / " + scan_date

    fig2.update_layout(
        height=1414,
        width=1000,
        showlegend=False,
        plot_bgcolor='white',
        title_text=title,
        # title_font_family="Arial Black",
    )

    fig2.update_xaxes(visible=False)
    fig2.update_yaxes(visible=False)

    fig2.show()

    fig2.write_image(volumetry_path + "/" + timestamp + "-report2.png")

    # img = io.imread(volumetry_path + "/" + timestamp + '-out-rad-grey.png')
    # figm1 = px.imshow(img)
    #
    # img = io.imread(volumetry_path + "/" + timestamp + '-out-rad-with-lab.png')
    # figm2 = px.imshow(img)
    #
    # # img = io.imread('out-rad-with-lab-axial.png')
    # # figm3 = px.imshow(img)
    # # img = io.imread('out-rad-with-lab-coronal.png')
    # # figm4 = px.imshow(img)
    #
    # fig = make_subplots(
    #     rows=4, cols=1, horizontal_spacing=0.01,
    #     vertical_spacing=0.001,
    #     specs=[[{"type": "image"}],
    #            [{"type": "image"}],
    #            [{"type": "image"}],
    #            [{"type": "table"}],
    #            #  [{"type": "table"}],
    #
    #            #  [{"type": "image"}]
    #            ])
    #
    # fig.add_trace(figm1.data[0], row=1, col=1)
    #
    # # fig.add_trace(figm2.data[0], row=3, col=1)
    #
    # fig.add_trace(figm2.data[0], row=2, col=1)
    #
    # # fig.add_trace(figm4.data[0], row=5, col=1)
    #
    # img = io.imread(volumetry_path + "/" + timestamp + '-fetus_3D.png')
    # figm3 = px.imshow(img)
    #
    # fig.add_trace(figm3.data[0], row=3, col=1)
    #
    # fig.add_trace(
    #     go.Table(header=dict(font_size=14, values=['Segmentation ROI', 'Volume [cc]']),
    #              cells=dict(fill_color='white', font_size=14, line_color='lightgray',
    #                         values=[["Fetus", "Placenta", "Amniotic fluid", "EFW (Baker)", "EFW (Kacem)"],
    #                                 [fetus, placenta, amniotic, baker, kacem]])),
    #     row=4, col=1
    # )
    #
    # # fig.add_trace(
    # #     go.Table(cells=dict(fill_color='white', font_size=14, line_color='lightgray',
    # #                             values=[["ID", "Date", "GA" ],
    # #                   [id, scan_date , ga ]])),
    # #     row=4, col=1
    # # )
    #
    # fig.update_xaxes(visible=False)
    # fig.update_yaxes(visible=False)
    #
    # # title = "Internal uterus volumetry: " + id + " / " + str(ga) + " weeks / " + scan_date
    # title = "Internal uterus volumetry: " + scan_date
    #
    # fig.update_layout(
    #     height=1414,
    #     # height=1200,
    #     width=1000,
    #     showlegend=False,
    #     plot_bgcolor='white',
    #     title_text=title,
    #     # title_font_family="Arial Black",
    # )
    #
    # fig.show()
    #
    # fig.write_image(volumetry_path + "/" + timestamp + "-report1.png")

    import img2pdf
    from PIL import Image

    import PyPDF2

    f_name_summary_intro = volumetry_path + "/" + timestamp + "-report1.png"
    f_name_summary_graphs = volumetry_path + "/" + timestamp + "-report2.png"

    # f_name_summary_intro = proc_dir + '/out-summary-intro.png'

    f_name_summary_intro_pdf = volumetry_path + "/" + timestamp + '-test-summary-intro.pdf'
    f_name_summary_graphs_pdf = volumetry_path + "/" + timestamp + '-test-summary-graphs.pdf'

    image = Image.open(f_name_summary_intro)
    pdf_bytes = img2pdf.convert(image.filename)
    file = open(f_name_summary_intro_pdf, "wb")
    file.write(pdf_bytes)
    image.close()
    file.close()

    # f_name_vol_centiles = proc_dir + '/out-volume-centiles.png'
    # f_name_vol_centiles_pdf = proc_dir + '/out-volume-centiles.pdf'

    image = Image.open(f_name_summary_graphs)
    pdf_bytes = img2pdf.convert(image.filename)
    file = open(f_name_summary_graphs_pdf, "wb")
    file.write(pdf_bytes)
    image.close()
    file.close()

    output_report_name_pdf = volumetry_path + "/" + timestamp + '-out-report-combined.pdf'

    pdf_merger = PyPDF2.PdfMerger()

    pdf_merger.append(f_name_summary_intro_pdf)
    pdf_merger.append(f_name_summary_graphs_pdf)
    with open(output_report_name_pdf, 'wb') as f:
        pdf_merger.write(f)

    # f_name_summary_intro = volumetry_path + "/" + timestamp + "-report1.png"
    #
    # # f_name_summary_intro = proc_dir + '/out-summary-intro.png'
    # f_name_summary_intro_pdf = volumetry_path + "/" + timestamp + '-test-summary-intro.pdf'
    # f_name_summary_intro_pdf_ = ("/home/sn21/freemax-transfer/Sara/landmarks-interface-autoplan/" + timestamp +
    #                              '-test-summary-intro.pdf')
    #
    # image = Image.open(f_name_summary_intro)
    # pdf_bytes = img2pdf.convert(image.filename)
    # file = open(f_name_summary_intro_pdf, "wb")
    # file.write(pdf_bytes)
    # image.close()
    # file.close()
    #
    # image = Image.open(f_name_summary_intro)
    # pdf_bytes = img2pdf.convert(image.filename)
    # file = open(f_name_summary_intro_pdf_, "wb")
    # file.write(pdf_bytes)
    # image.close()
    # file.close()
    #
    # # f_name_vol_centiles = proc_dir + '/out-volume-centiles.png'
    # # f_name_vol_centiles_pdf = proc_dir + '/out-volume-centiles.pdf'
    #
    # # image = Image.open(f_name_vol_centiles)
    # # pdf_bytes = img2pdf.convert(image.filename)
    # # file = open(f_name_vol_centiles_pdf, "wb")
    # # file.write(pdf_bytes)
    # # image.close()
    # # file.close()
    #
    # # #f_name_report_pdf = proc_dir + '/out-report-combined.pdf'
    #
    # # merge_pdfs(f_name_summary_intro_pdf, f_name_vol_centiles_pdf, output_report_name_pdf)

    print()
    print("--------------------------------------------------------------")
    print()

    # INCLUDE THIS SCRIPT IN THE FIRE INITIALIZATION

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
