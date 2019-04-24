# convert masks from geojson
# convert geojson from masks
import os
from skimage import io
# import segmentationUtils
# import annotationUtils
from imgseg import segmentationUtils
from imgseg import annotationUtils
from geojson import FeatureCollection, dump
from skimage import measure

def gen_mask_from_geojson(files_proc, img_size=None, infer=False):
    # %% Some housekeeping to setup example data
    # files_proc= [os.path.abspath(os.path.join('..','data','maskGenerator','img','annotation.json'))]

    # masks_to_create = {
    #   "cells": ['filled', 'edge', 'distance', 'weigthed'],
    #   "nuclei": ['filled', 'edge', 'distance', 'weigthed'],
    # }
    masks_to_create = {}
    masks_to_create_value = ['filled', 'edge', 'distance', 'weigthed']

    # annot_types = list(masks_to_create.keys())

    annotationsImporter = annotationUtils.GeojsonImporter()

    # Instance to save masks
    masks = annotationUtils.MaskGenerator()

    weightedEdgeMasks = annotationUtils.WeightedEdgeMaskGenerator(sigma=8, w0=10)
    distMapMasks = annotationUtils.DistanceMapGenerator(truncate_distance=None)

    #%% Loop over all files
    for file_proc in files_proc:
        print('PROCESSING FILE:')
        print(file_proc)

        # Decompose file name
        drive, path_and_file = os.path.splitdrive(file_proc)
        path, file = os.path.split(path_and_file)
        file_base, ext = os.path.splitext(file)

        # Read annotation:  Correct class has been selected based on annot_type
        annot_dict_all, roi_size_all, image_size = annotationsImporter.load(file_proc)
        if img_size is not None:
            image_size = img_size

        annot_types = set(annot_dict_all[k]['properties']['label'] for k in annot_dict_all.keys())
        print("annot_types: ", annot_types)

        for annot_type in annot_types:
            print("annot_type: ", annot_type)
            masks_to_create[annot_type] = masks_to_create_value

            # Filter the annotations by label
            annot_dict = {k: annot_dict_all[k] for k in annot_dict_all.keys() if annot_dict_all[k]['properties']['label'] == annot_type}

            # Create masks

            # Binary - is always necessary to creat other masks
            print(' .... creating binary masks .....')
            binaryMasks = annotationUtils.BinaryMaskGenerator(image_size=image_size, erose_size=5, obj_size_rem=500, save_indiv=True)
            mask_dict = binaryMasks.generate(annot_dict)

            # Save binary masks FILLED if specified
            if 'filled' in masks_to_create[annot_type]:
                if infer:
                    file_name_save = os.path.join(drive,path, annot_type + '_filled_output.png')
                else:
                    file_name_save = os.path.join(drive,path, annot_type + '_filled.png')
                masks.save(mask_dict,'fill',file_name_save)

            # # Edge mask
            # if 'edge' in masks_to_create[annot_type]:
            #     if infer:
            #         file_name_save = os.path.join(drive,path, annot_type + '_edge_output.png')
            #     else:
            #         file_name_save = os.path.join(drive,path, annot_type + '_edge.png')
            #     masks.save(mask_dict,'edge',file_name_save)

            # Distance map
            if 'distance' in masks_to_create[annot_type]:
                print(' .... creating distance maps .....')
                mask_dict = distMapMasks.generate(annot_dict,mask_dict)

                # Save
                if infer:
                    file_name_save = os.path.join(drive,path, annot_type + '_distmap_output.png')
                else:
                    file_name_save = os.path.join(drive,path, annot_type + '_distmap.png')
                masks.save(mask_dict,'distance_map',file_name_save)

            # # Weighted edge mask
            # if 'weigthed' in masks_to_create[annot_type]:
            #     print(' .... creating weighted edge masks .....')
            #     mask_dict = weightedEdgeMasks.generate(annot_dict,mask_dict)
            #
            #     # Save
            #     if infer:
            #         file_name_save = os.path.join(drive,path, annot_type + '_edgeweight_output.png')
            #     else:
            #         file_name_save = os.path.join(drive,path, annot_type + '_edgeweight.png')
            #     masks.save(mask_dict,'edge_weighted',file_name_save)


def masks_to_annotation(outputs_dir):
    # %% Process one folder and save as one json file allowing multiple annotation types
    simplify_tol = 1  # Tolerance for polygon simplification with shapely (0 to not simplify)

    # outputs_dir = os.path.abspath(os.path.join('..', 'data', 'postProcessing', 'mask2json'))
    if os.path.exists(outputs_dir):
        print(f'Analyzing folder:{outputs_dir}')
        features = []  # For geojson
        image_size = None
        for file in [f for f in os.listdir(outputs_dir) if '_filled_output.png' in f]:
            # Read png with mask
            print(f'Analyzing file:{file}')

            file_full = os.path.join(outputs_dir, file)
            mask_img = io.imread(file_full)
            print("mask_img.shape:", mask_img.shape)
            mask = measure.label(mask_img)

            # Here summarizing the geojson should occur
            image_size = mask_img.shape  # This might cause problems if any kind of binning was performed

            # Get label from file name
            label = file.split('_filled_output.png', 1)[0]
            print("label:", label)
            # print(mask_img[0:1, :100])

            # Call function to transform segmentation masks into (geojson) polygons
            feature, contours = segmentationUtils.masks_to_polygon(mask,
                                                                    label=label,
                                                                    simplify_tol=simplify_tol)
                                                                    # save_name=file_full.replace(".png", ".json"))
            features.append(feature)

        feature_collection = FeatureCollection(features, bbox=[0, 0.0, image_size[0], image_size[1]])

        # Save to json file
        save_name_json = os.path.join(outputs_dir, 'prediction.json')
        with open(save_name_json, 'w') as f:
            dump(feature_collection, f)
            f.close()


if __name__ == "__main__":
    print(os.getcwd())
    # generate mask from annotation.json
    files_proc = ["datasets/anet_png/train/w1_bac_kif1c_6512_p02_/annotation.json"]
    gen_mask_from_geojson(files_proc=files_proc)

    # # generate mask from prediction.json
    # files_proc = ["datasets/anet_png/test/w11_bac_bora_4437_p14_/prediction.json"]
    # # files_proc = ["datasets/anet_png/test/w11_bac_bora_4437_p14_/annotation.json"]
    # # gen_mask_from_geojson(files_proc=files_proc, infer=True)
    # gen_mask_from_geojson(files_proc=files_proc, infer=False)
    #
    #
    # # outputs_dir = 'datasets/anet_png/test/w11_bac_bora_4437_p14_/'
    # # masks_to_annotation(outputs_dir)
