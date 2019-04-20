# Instances to import annotations
import os
import annotationUtils


def generate_mask(files_proc,
                  image_size=(925, 621),
                  annot_type='geojson'):

    # files_proc= ['/home/alex/ImJoyWorkspace/default/Anet-Lite/datasets/example/train/z010/annotation.json']
    channels = [{"identifier": 'annotation', "name": 'cell', "masks": ['filled', 'edge', 'distance', 'weigthed']}]
    # annot_type = 'geojson'
    # labels = ['cell']
    # image_size = (925, 621)

    if annot_type == 'fiji':
        annotationsImporter = annotationUtils.FijiImporter()
    elif annot_type == 'geojson':
        annotationsImporter = annotationUtils.GeojsonImporter()

    # Instance to save masks
    masks = annotationUtils.MaskGenerator()

    # Instances to to create masks
    binaryMasks = annotationUtils.BinaryMaskGenerator(image_size = image_size, erose_size=5, obj_size_rem=500, save_indiv=True)
    weightedEdgeMasks = annotationUtils.WeightedEdgeMaskGenerator(sigma=8, w0=10)
    distMapMasks = annotationUtils.DistanceMapGenerator(truncate_distance=None)

    # Transform channel list in dictionary
    channels_new = {}
    for iter, dic in enumerate(channels):
        channels_new[dic["identifier"]] = {}

        channels_new[dic["identifier"]]['name'] = dic["name"]
        channels_new[dic["identifier"]]['masks'] = dic["masks"]

    channel_ident = list(channels_new.keys())
    # Loop over all files
    for file_proc in files_proc:
        print('PROCESSING FILE:')
        print("file_proc:", file_proc)

        # Decompose file name
        drive, path_and_file = os.path.splitdrive(file_proc)
        path, file = os.path.split(path_and_file)
        file_base, ext = os.path.splitext(file)

        # print("drive:", drive)
        # print("path_and_file:", path_and_file)
        # print("path:", path)
        # print("file:", file)
        # print("file_base:", file_base)
        # print("ext:", ext)

        print("channel_ident:", channel_ident)

        # Check which channel this is
        #  [ToDo]: Not perfect since it returns the first hit.
        file_ch = next((substring for substring in channel_ident if substring in file_base), None)

        print("file_ch:", file_ch)

        if not file_ch:
            print(f'No channel identifier found in file name {file_base}')
            continue

        print(f'Mask type identified: {file_ch}')

        # Read annotation:  Correct class has been selected based on annot_type
        annot_dict_all, roi_size_all = annotationsImporter.load(file_proc)

        labels = list(set([annot_dict_all[k]['properties']['label'] for k in annot_dict_all.keys()]))
        print("labels:", labels)
        for label in labels:

            # Filter the annotations by label
            annot_dict = {k: annot_dict_all[k] for k in annot_dict_all.keys() if annot_dict_all[k]['properties']['label'] == label}
            # Create masks

            # Binary - is always necessary to creat other masks
            print(' .... creating binary masks .....')
            mask_dict = binaryMasks.generate(annot_dict)
            # print("annot_dict:", annot_dict)
            # print("mask_dict:", mask_dict)
            print("channels_new:", channels_new)

            # Save binary masks FILLED if specified
            if 'filled' in channels_new[file_ch]['masks']:

                file_name_save = os.path.join(drive,path, file_base + '__MASK_fill.png')
                masks.save(mask_dict,'fill',file_name_save)

            # # Edge mask
            # if 'edge' in channels_new[file_ch]['masks']:
            #     file_name_save = os.path.join(drive,path, file_base + '__MASK_edge.png')
            #     masks.save(mask_dict,'edge',file_name_save)

            # Distance map
            if 'distance' in channels_new[file_ch]['masks']:
                print(' .... creating distance maps .....')
                mask_dict    = distMapMasks.generate(annot_dict,mask_dict)

                # Save
                file_name_save = os.path.join(drive,path, file_base + '__MASK_distMap.png')
                masks.save(mask_dict,'distance_map',file_name_save)


            # # Weighted edge mask
            # if 'weigthed' in channels_new[file_ch]['masks']:
            #     print(' .... creating weighted edge masks .....')
            #     mask_dict = weightedEdgeMasks.generate(annot_dict,mask_dict)
            #
            #     # Save
            #     file_name_save = os.path.join(drive,path, file_base + '__MASK_edgeWeight.png')
            #     masks.save(mask_dict,'edge_weighted',file_name_save)
