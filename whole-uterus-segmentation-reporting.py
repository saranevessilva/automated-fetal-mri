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


date_path = datetime.today().strftime("%Y-%m-%d")
volumetry_path = debugFolder + "/" + date_path

# Find the latest added .nii.gz files
files = sorted(glob.glob(os.path.join(volumetry_path, "**/*.nii.gz"), recursive=True), key=os.path.getmtime, reverse=True)

# Get the latest added file's directory
latest = files[0]
latest_path = os.path.dirname(latest)
timestamp = latest_path.split("/")[-1].split("-")[:3]  # Extracts ['HH', 'MM', 'SS']
timestamp = "-".join(timestamp)  # Reconstruct timestamp as 'HH-MM-SS'


# Run prediction with nnUNet
# Set the DISPLAY and XAUTHORITY environment variables
os.environ['DISPLAY'] = ':0'  # Replace with your X11 display, e.g., ':1.0'
os.environ["XAUTHORITY"] = '/opt/code/automated-fetal-mri/.Xauthority'

# Record the start time
start_time = time.time()

# Define the terminal command for prediction
terminal_command = (("export nnUNet_raw='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_raw'; export "
                     "nnUNet_preprocessed='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_preprocessed'; "
                     "export nnUNet_results='/opt/code/automated-fetal-mri/volumetry/Volumetry/nnUNet_results'; "
                     "nnUNetv2_predict -i ") + volumetry_path + "/"
                    + timestamp + "-nnUNet_seg-volumetry/ -o " + volumetry_path + "/" + timestamp
                    + "-nnUNet_pred-volumetry/ -d 084 -c 3d_fullres -f 1")

print("Executing command:", terminal_command)
os.system(terminal_command)

# Run the terminal command
subprocess.run(terminal_command, shell=True)

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

file_ga = debugFolder + "ga.txt"

file_id = debugFolder + "id.txt"

volumetry_path = debugFolder + "/" + date_path

# load .nii images in NiBabel format

img_name = volumetry_path + "/" + timestamp + "-nnUNet_seg-volumetry/Volumetry_001_0000.nii.gz"

lab_name = segmentation_filename

img_nii = nib.load(img_name)
lab_nii = nib.load(lab_name)

# extract matrices only
img_raw = img_nii.get_fdata()
lab_raw = lab_nii.get_fdata()
#
# ga = 37.43

with open(file_ga, 'r') as file:
    content = file.read()

    # Search for the line containing 'ga' in the format 'weeks+days'
    match = re.search(r'(\d+)\+(\d+)', content)
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

# Read the file and store the content in 'id'
with open(file_id, 'r') as file:
    id = file.read().strip()  # Read the file and remove any surrounding whitespace/newlines

# Print the extracted ID value
if id:
    print(f"Extracted ID value: {id}")
else:
    print("ID value not found.")

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

print()
print("--------------------------------------------------------------")
print()
