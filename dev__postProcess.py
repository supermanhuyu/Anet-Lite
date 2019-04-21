
import importlib
import sys
import os
from skimage import io
import numpy as np
import segmentationUtils
importlib.reload(segmentationUtils)


# outputs_dir = 'datasets/anet_png/train/w11_bac_bora_4437_p02_/'
# # outputs_dir = 'datasets/example/train/z010/'
# simplify_tol = 0  # Tolerance for polygon simplification with shapely (0 to not simplify)
#
# if os.path.exists(outputs_dir):
#     for file in [f for f in os.listdir(outputs_dir) if 'MASK' in f][:1]:
#         file = os.path.join(outputs_dir, file)
#         # Read png with input and output
#         mask = io.imread(file)
#         print("file:",file)
#         print("mask.shape:", mask.shape)
#         # Call function to transform segmentation masks into (geojson) polygons
#         segmentationUtils.masks_to_polygon(mask,
#                                            simplify_tol=simplify_tol,
#                                            plot_simplify=False,
#                                            save_name=file
#                                            .replace('.png', '.json')
#                                            .replace('MASK', 'jeojson'))
#
#         print("segmentationUtils convert success")


def masks_to_annotation(filepath, save_name, simplify_tol=0, plot_simplify=False):
    mask = io.imread(filepath)
    contours, feature_collection = segmentationUtils.masks_to_polygon(mask,
                                                                      simplify_tol=simplify_tol,
                                                                      plot_simplify=plot_simplify,
                                                                      save_name=save_name)
    return feature_collection
