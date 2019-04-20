from generate_mask_geojson import generate_mask
import os
from skimage import io
# image_size = (2048, 2048)

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
# files_proc = ["datasets/example/train/z010/annotation__jeojson_fill.json"]


import os
import shutil
from PIL import Image

sample_dir = "datasets/anet/"
# train_dir = "datasets/anet/train"
train_dir = "datasets/anet/valid"

sample_list = os.listdir(train_dir)
for sample in sample_list:
    tiff_list = os.listdir(os.path.join(train_dir, sample))
    for file in tiff_list:
        if file.endswith(".tif"):
            print("sample:", sample)
            print("file:", file)
            tif = Image.open(os.path.join(train_dir, sample, file)).convert(mode='I')
            # print(type(tif))
            # print(tif.size)
            save_dir = os.path.dirname(os.path.join(train_dir, sample, file).replace("anet", "anet_png"))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = os.path.join(train_dir, sample, file).replace("anet", "anet_png").replace(".tif", ".png")
            print("save_dir:", save_dir)
            print("save_file:", save_file)
            tif.save(save_file)
