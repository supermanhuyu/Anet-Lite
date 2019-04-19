# Code to post-process segmentation results
#  Part about actual cell segmentation comes from https://gitlab.pasteur.fr/wouyang/im2im-segmentation:
#    ==> run_im2im.py --> calls segment_cells_nuclei (from im2imLib.segmentationUtils)

# %% Test files
#  1. Change to local directory
#  2. Test data are on GitHub: img-segment/data/postprocessing/outputs/data/postprocessing/cells_nuclei
outputs_dir = 'datasets/example/train/z010/'

# %% Test modules
import importlib  # to reload: importlib.reload(AnnotationImporter
import sys
import os
from skimage import io
import numpy as np

# sys.path.append('/Volumes/PILON_HD2/fmueller/Documents/code/ImJoy_dev/img-segment/imgseg')

# %% Loop over files and perform postprocessing

import segmentationUtils

importlib.reload(segmentationUtils)

h_threshold = 15  # 15; morphological depth for nuclei separation (watershed)
min_size_cell = 20  # 200; minimum size of cell
min_size_nuclei = 100  # 1000; minimum size of the nuclei
# skimage.morphology.remove_small_objects
simplify_tol = 1  # Tolerance for polygon simplification with shapely (0 to not simplify)

if os.path.exists(outputs_dir):
    for file in [f for f in os.listdir(outputs_dir) if 'MASK' in f]:
        file = os.path.join(outputs_dir, file)

        # Read png with input and output
        output_png = io.imread(file)
        # input_png = io.imread(file.replace('outputs.png', 'inputs.png'))

        # Create empty arrays to store the relevant channels for postprocessing
        img_output = np.zeros((output_png.shape[0], output_png.shape[1], 2))
        # img_input = np.zeros((output_png.shape[0], output_png.shape[1], 1))

        img_output[:, :, 0] = output_png[:, :, 0]  # Cell mask
        img_output[:, :, 1] = output_png[:, :, 2]  # Nuclei mask
        print("img_output.shape:", img_output.shape)

        # img_output[:, :, 0] = input_png[:, :, 0]  # Image of cell

        # Call function to perform segmentation
        cytoplasm_mask, nuclei_mask = segmentationUtils.segment_cells_nuclei(img_output,
                                                                             img_output,
                                                                             h_threshold=h_threshold,
                                                                             min_size_cell=min_size_cell,
                                                                             min_size_nuclei=min_size_nuclei,
                                                                             save_path=file.replace('.png', '_'))

        print("cytoplasm_mask.shape:", cytoplasm_mask.shape)
        print("nuclei_mask.shape:", nuclei_mask.shape)

        # Call function to transform segmentation masks into (geojson) polygons
        segmentationUtils.masks_to_polygon(cytoplasm_mask,
                                           simplify_tol=simplify_tol,
                                           plot_simplify=False,
                                           save_name=file.replace('.png', '__cells_polygon.json'))

        segmentationUtils.masks_to_polygon(nuclei_mask,
                                           simplify_tol=simplify_tol,
                                           plot_simplify=False,
                                           save_name=file.replace('.png', '__nuclei_polygon.json'))
        print("segmentationUtils convert success")
