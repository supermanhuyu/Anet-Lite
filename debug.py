from generate_mask_geojson import generate_mask
import os
from skimage import io
image_size = (2048, 2048)

# train_list = ["w11_bac_bora_4437_p02_",
#               "w11_bac_bora_4437_p13_",
#               "w11_bac_bora_4437_p30_",
#               "w11_bac_tubggcp3_4430_p04_",
#               "w13_bac_usf1_7394_p02_",
#               "w1_bac_kif1c_6512_p02_"]
#
# files_proc = ["datasets/anet/train/w1_bac_kif1c_6512_p02_/annotation.json",
#               "datasets/anet/train/w11_bac_bora_4437_p13_/annotation.json",
#               "datasets/anet/train/w11_bac_bora_4437_p30_/annotation.json",
#               "datasets/anet/train/w11_bac_tubggcp3_4430_p04_/annotation.json",
#               "datasets/anet/train/w13_bac_usf1_7394_p02_/annotation.json",]

# generate_mask(files_proc=files_proc, image_size=image_size)

# files_proc = ["datasets/anet/train/w11_bac_bora_4437_p02_/annotation__jeojson_fill.json"]
files_proc = ["datasets/example/train/z010/annotation__jeojson_fill.json"]
# # image_size = (624, 924)
# generate_mask(files_proc=files_proc, image_size=image_size)

outputs_dir = os.path.dirname(files_proc[0])
print(outputs_dir)


if os.path.exists(outputs_dir):
    for file in [f for f in os.listdir(outputs_dir) if 'MASK' in f]:
        file = os.path.join(outputs_dir, file)
        # Read png with input and output
        mask = io.imread(file)
        print("mask.shape:", mask.shape)
