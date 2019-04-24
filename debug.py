# # generate mask from geojson
# from generate_mask_geojson import gen_mask_from_geojson
# import os
# from skimage import io
# image_size = (2048, 2048)
# files_proc = ['datasets/anet_png/test/w11_bac_tubggcp3_4430_p12_/annotation.json']
# gen_mask_from_geojson(files_proc=files_proc, image_size=image_size)

# generate annotaion from mask



# convert tif to png
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
        full_file = os.path.join(train_dir, sample, file)
        if file.endswith(".tif"):
            print("sample:", sample)
            print("full_file:", full_file)
            tif = Image.open(full_file).convert(mode='I')
            # print(type(tif))
            # print(tif.size)
            save_dir = os.path.dirname(full_file.replace("anet_tif", "anet_png"))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = full_file.replace("anet_tif", "anet_png").replace(".tif", ".png")
            print("save_dir:", save_dir)
            print("save_file:", save_file)
            tif.save(save_file)
        else:
            shutil.copy(full_file, full_file.replace("anet_tif", "anet_png"))
