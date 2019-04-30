# Intended as an one time conversion of FiJi annotations to ImJoy's annotation
#  format. 

#%% Import modules
import os
import shutil
import numpy as np
from read_roi import read_roi_zip  # https://github.com/hadim/read-roi
from geojson import Polygon as geojson_polygon
from geojson import Feature, FeatureCollection, dump  # Used to create and save the geojson files: pip install geojson

# Create folders
def create_folder(folder_new):
    if not os.path.isdir(folder_new):
        os.makedirs(folder_new)


#%% Some parameters
annot_ext = '_ROI.zip'
img_ext='.tif'
path_open = 'datasets/fijiroi'  # Data is also on dropbox
image_size = (2048,2048)


#%% Recursive search to find all files
files_proc = []

for root, dirnames, filenames in os.walk(path_open):
    # print("root:", root)
    # print("dirnames:", dirnames)
    # print("filenames:", filenames)
    for filename in filenames:
        if filename.endswith(annot_ext):
            files_proc.append(os.path.join(root, filename))


#%% Loop over all files
            
ident_ch1 = 'Cy5'
ident_ch2 = 'DAPI'
print("files_proc:", files_proc)

for file_proc in files_proc:

    print(f'PROCESSING FILE: {file_proc}')

    # Decompose file name
    drive, path_and_file = os.path.splitdrive(file_proc)
    path, file = os.path.split(path_and_file)
    file_base = file.replace(annot_ext,'')

    if not ident_ch1 in file_base:
        print(f'No channel identifier found in file name {file_base}')
        continue

    # Create sub-folder and remove channel identifier
    subfolder = file_base.replace(ident_ch1, "")
    folder_save = os.path.join(drive,path,'_anet',subfolder)
    create_folder(folder_save)

    # Open ROI file
    roi_dict_complete = read_roi_zip(file_proc)
    
    features = []   # For geojson
    for key_roi, val_roi in roi_dict_complete.items():

        # Get coordinates - maybe x and y have to be exchanged
        # pos = np.column_stack((val_roi['y'], val_roi['x']))
        pos = np.column_stack((val_roi['x'], [image_size[1] - h for h in val_roi['y']]))

        # Create and append feature for geojson
        pol_loop = geojson_polygon([pos.tolist()])
        features.append(Feature(geometry=pol_loop,properties= {"label": 'cells'})) #,  properties={"country": "Spain"}) #)

    # Open ROI file
    roi_dict_complete = read_roi_zip(file_proc.replace(ident_ch1,ident_ch2))
    
    for key_roi, val_roi in roi_dict_complete.items():

        # Get coordinates - maybe x and y have to be exchanged
        # pos = np.column_stack((val_roi['y'], val_roi['x']))
        pos = np.column_stack((val_roi['x'], [image_size[1] - h for h in val_roi['y']]))

        # Create and append feature for geojson
        pol_loop = geojson_polygon([pos.tolist()])
        features.append(Feature(geometry=pol_loop,properties= {"label": 'nuclei'})) #,  properties={"country": "Spain"}) #)

    # Create geojson feature collection
    feature_collection = FeatureCollection(features,bbox = [0, 0, image_size[0], image_size[1]])

    # Save to json file
    save_name_json = os.path.join(folder_save, 'annotation.json')
    with open(save_name_json, 'w') as f:
        dump(feature_collection, f)
        f.close()

    # Find and copy raw data renamed with channel identifier
    img_raw = os.path.join(drive,path,file_base+img_ext)
    if os.path.isfile(img_raw):
        img_raw_new = os.path.join(folder_save, 'cells'+img_ext)
        shutil.copy(img_raw, img_raw_new)
        print(f'Copying raw image: {img_raw}')

    else:
        print(f'Raw image does not exist: {img_raw}')
        
    # Find and copy raw data renamed with channel identifier
    img_raw = os.path.join(drive,path,file_base.replace(ident_ch1,ident_ch2)+img_ext)
    if os.path.isfile(img_raw):
        img_raw_new = os.path.join(folder_save, 'nuclei'+img_ext)
        shutil.copy(img_raw, img_raw_new)
        print(f'Copying raw image: {img_raw}')

    else:
        print(f'Raw image does not exist: {img_raw}')
