<script lang="python">
import os
import sys
import requests
import zipfile
import random
import string
import asyncio
import numpy as np
from PIL import Image
from scipy import misc
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.models import load_model
import base64
from io import BytesIO
import tensorflow as tf
import json
from keras import backend as K
import threading
import shutil
import time
from skimage import io, measure
from geojson import FeatureCollection, dump

os.chdir('Anet-Lite')
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.examples import TransformedTubulin001
from anet.data.file_loader import ImageLoader
from anet.data.utils import make_generator, make_test_generator
from anet.networks import UnetGenerator, get_dssim_l1_loss
from anet.utils import export_model_to_js
from imgseg import segmentationUtils
from imgseg import annotationUtils
from imgseg import DRFNStoolbox

# import importlib
# importlib.reload(UnetGenerator)

abort = threading.Event()


def plot_tensors(dash, tensor_list, label, titles):
    image_list = [tensor.reshape(tensor.shape[-3], tensor.shape[-2], -1) for tensor in tensor_list]
    displays = {}

    for i in range(len(image_list)):
        ims = image_list[i]
        for j in range(ims.shape[2]):
            im = ims[:, :, j]
            min = im.min()
            im = Image.fromarray(((im - min) / (im.max() - min) * 255).astype('uint8'))
            buffered = BytesIO()
            im.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
            imgurl = 'data:image/png;base64,' + img_str
            displays[titles[i][j]] = imgurl
        dash.appendDisplay(label, displays)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


class UpdateUI(Callback):
    def __init__(self, total_epoch, dash, gen, opt):
        self.total_epoch = total_epoch
        self.epoch = 0
        self.logs = {}
        self.dash = dash
        self.step = 0
        self.gen = gen
        self.input_channels = [ch[0] for ch in opt.input_channels]
        self.target_channels = [ch[0] for ch in opt.target_channels]
        self.output_channels = ['output_' + ch[0] for ch in opt.target_channels]

    def on_batch_end(self, batch, logs):
        self.logs = logs
        api.showStatus('training epoch:' + str(self.epoch) + '/' + str(self.total_epoch) + ' ' + str(logs))
        sys.stdout.flush()
        self.dash.updateCallback('onStep', self.step, {'mse': np.asscalar(logs['mean_squared_error']),
                                                       'dssim_l1': np.asscalar(logs['DSSIM_L1'])})
        self.step += 1
        if abort.is_set():
            raise Exception('Abort.')

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        self.logs = logs
        api.showProgress(self.epoch / self.total_epoch * 100)
        api.showStatus('training epoch:' + str(self.epoch) + '/' + str(self.total_epoch) + ' ' + str(logs))
        xbatch, ybatch = next(self.gen)
        ypbatch = self.model.predict(xbatch, batch_size=1)
        tensor_list = [xbatch, ypbatch, ybatch]
        label = 'Step ' + str(self.step)
        titles = [self.input_channels, self.output_channels, self.target_channels]
        plot_tensors(self.dash, tensor_list, label, titles)

class my_config():
    def __init__(self, name="", epochs=20, batchsize=2, steps=10):
        self.name = name
        self.epochs = epochs
        self.steps = steps
        self.batchsize = batchsize

def my_opt(config):
    # work_dir = os.path.join("datasets", config["samples"][0]["data"][0].split("/")[2])
    if config["root_folder"].startswith("/"):
        work_dir = "datasets" + config["root_folder"]
    else:
        work_dir = config["root_folder"]
    opt = Options().parse(['--work_dir={}'.format(work_dir)])
    opt.work_dir = work_dir
    opt.input_size = 256
    opt.base_filter = 4
    opt.input_channels = []
    opt.target_channels = []
    # opt.load_from = None
    # opt.checkpoints_dir = opt.work_dir + "/__model__"

    opt.channel = config.get("channel_config")
    for key in opt.channel.keys():
        opt.input_channels.append((opt.channel[key]["name"], {'filter': "*"+opt.channel[key]["filter"]+"*", 'loader': ImageLoader()},))

    try:
        annotation_types = config.get("annotation_types")
        for key in annotation_types.keys():
            print("get label:", annotation_types[key].get("label"))
            opt.target_channels.append((annotation_types[key].get("label")+"_filled", {'filter': "*"+annotation_types[key].get("label")+"_filled*", 'loader': ImageLoader()},))
            opt.target_channels.append((annotation_types[key].get("label")+"_distmap", {'filter': "*"+annotation_types[key].get("label")+"_distmap*", 'loader': ImageLoader()},))
    except:
        print("get label error from annotation_types in the config.json.")
        pass

    opt.input_nc = len(opt.input_channels)
    opt.target_nc = len(opt.target_channels)
    return opt


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
        # if img_size is not None:
        #     image_size = img_size

        annot_types = set(annot_dict_all[k]['properties']['label'] for k in annot_dict_all.keys())
        print("annot_types: ", annot_types)

        for annot_type in annot_types:
            # print("annot_type: ", annot_type)
            masks_to_create[annot_type] = masks_to_create_value

            # Filter the annotations by label
            annot_dict = {k: annot_dict_all[k] for k in annot_dict_all.keys() if annot_dict_all[k]['properties']['label'] == annot_type}
            # print("len(annot_dict):", len(annot_dict))
            # print("annot_dict.keys():", annot_dict.keys())

            # Create masks

            # Binary - is always necessary to creat other masks
            print(' .... creating binary masks .....')
            binaryMasks = annotationUtils.BinaryMaskGenerator(image_size=img_size, erose_size=5, obj_size_rem=500, save_indiv=True)
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
    simplify_tol = 0  # Tolerance for polygon simplification with shapely (0 to not simplify)

    # outputs_dir = os.path.abspath(os.path.join('..', 'data', 'postProcessing', 'mask2json'))
    if os.path.exists(outputs_dir):
        print(f'Analyzing folder:{outputs_dir}')
        features = []  # For geojson
        image_size = None
        for file in [f for f in os.listdir(outputs_dir) if '_distmap_output.png' in f]:
            # Read png with mask
            print(f'Analyzing file:{file}')

            file_full = os.path.join(outputs_dir, file)
            mask_img = io.imread(file_full)
            print("mask_img.shape:", mask_img.shape)
            mask = measure.label(mask_img)
            nuclei_mask = DRFNStoolbox.seedless_segment(mask, 15, p_thresh=0.5)
            img = Image.fromarray(nuclei_mask.astype('uint8'))
            img.save(os.path.join(outputs_dir, file.replace('.png', '_noSeeds_OBJECTS.png')))

            cell_mask = DRFNStoolbox.segment_with_seed(mask, nuclei_mask, 15, p_thresh=0.5)
            img = Image.fromarray(cell_mask.astype('uint8'))
            img.save(os.path.join(outputs_dir, file.replace('.png', '_wSeeds_OBJECTS.png')))

            # Here summarizing the geojson should occur
            image_size = mask_img.shape  # This might cause problems if any kind of binning was performed

            # Get label from file name
            label = file.split('_distmap_output.png', 1)[0]
            print("label:", label)
            # print(mask_img[0:1, :100])

            # Call function to transform segmentation masks into (geojson) polygons
            feature, contours = segmentationUtils.masks_to_polygon(cell_mask,
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
        return feature_collection

class ImJoyPlugin():
    def __init__(self):
        self._initialized = False

    def download_data(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before downloading.')
            return
        work_dir = self._opt.work_dir
        target_dir = os.path.normpath(os.path.join(work_dir, '../example_data_' + ''.join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(4))))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        url = 'https://www.dropbox.com/s/02w4he65f2cnf1t/EM_membrane_dataset.zip?dl=1'
        api.showStatus('downloading...')
        r = requests.get(url, allow_redirects=True)
        name_zip = os.path.join(work_dir, 'EM_membrane_dataset.zip')
        open(name_zip, 'wb').write(r.content)
        api.showStatus('unzipping...')
        with zipfile.ZipFile(name_zip, 'r') as f:
            f.extractall(target_dir)
        # os.remove(name_zip)
        api.showStatus('example data saved to ' + os.path.abspath(target_dir))
        self._opt.work_dir = target_dir

    def initialize(self, opt):
        self.model = UnetGenerator(input_size=opt.input_size, input_channels=opt.input_nc,
                                   target_channels=opt.target_nc, base_filter=opt.base_filter)
        if opt.load_from is not None:
            print('loading weights from: ' + opt.load_from)
            self.model.load_weights(opt.load_from)
        else:
            model_path = os.path.join(opt.checkpoints_dir, '__model__.hdf5')
            model_config_path = os.path.join(opt.work_dir, 'model_config.json')
            if os.path.exists(model_path) and os.path.exists(model_config_path):
                with open(model_config_path) as f:
                    model_config = json.load(f)

                if model_config['input_size'] == opt.input_size \
                        and model_config['input_channels'] == len(opt.input_channels) \
                        and model_config['target_channels'] == len(opt.target_channels):
                    print("model_config:", model_config)
                    self.model.load_weights(model_path)
                # self.model.load_weights(model_path)
        DSSIM_L1 = get_dssim_l1_loss()
        self.model.compile(optimizer='adam',
                           loss=DSSIM_L1,
                           metrics=['mse', DSSIM_L1])
        self._initialized = True
        # api.showStatus("A-net lite successfully initialized.")

    async def setup(self):
        # api.register(name="set working directory", run=self.set_work_dir, ui="set working directory for loading data and saving trained models")
        api.register(name="get example dataset", run=self.download_data,
                     ui="download example data set to your workspace")
        api.register(name="load trained weights", run=self.load_model_weights,
                     ui="load a trained weights for the model")
        api.register(name="train", run=self.train, ui="name:{id:'name', type:'string', placeholder:''}<br>" \
                                                      "epochs:{id:'epochs', type:'number', min:1, placeholder:100}<br>" \
                                                      "steps per epoch:{id:'steps', type:'number', min:1, placeholder:30}<br>" \
                                                      "batch size:{id:'batchsize', type:'number', min:1, placeholder:4}<br>"
                     )
        api.register(name="freeze and export model", run=self.freeze_model, ui="freeze and export the graph as pb file")
        api.register(name="train_run", run=self.train_run, ui="train_run")
        api.register(name="test_run", run=self.test_run, ui="test_run")
        api.register(name="test", run=self.test,
                     ui="number of images:{id:'num', type:'number', min:1, placeholder:10}<br>")

    async def load_model_weights(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before loading weights.')
            return
        lastPath = await api.getConfig('work_dir')
        try:
            weight_path = await api.showFileDialog(root=lastPath, type='file')
            if os.path.exists(weight_path):
                print('loading weights from: ' + weight_path)
                self.model.load_weights(weight_path)
                api.showStatus('weights loaded from ' + weight_path)
        except:
            pass

    def export(self, my):
        opt = self._opt
        export_model_to_js(self.model, opt.work_dir + '/__js_model__')

    async def run(self, my):
        await self.train_run("")
        pass

    async def test(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before testing.')
            return
        sources = GenericTransformedImages(self._opt)
        batch_size = 1
        source = sources['test']
        count = 0
        output_dir = os.path.join(self._opt.work_dir, 'outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gen = make_test_generator(source, batch_size=batch_size)
        api.showStatus('making predictions.')
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10,
                                           data={"display_mode": "all"})

        input_channels = [ch[0] for ch in self._opt.input_channels]
        output_channels = ['output_' + ch[0] for ch in self._opt.target_channels]

        for i in range(int(totalsize / batch_size + 0.5)):
            xbatch, paths = next(gen)
            ypbatch = self.model.predict(xbatch, batch_size=batch_size)
            tensor_list = [xbatch, ypbatch]
            label = 'Sample ' + str(i)
            titles = [input_channels, output_channels]
            plot_tensors(self.dash, tensor_list, label, titles)
            count += batch_size
            for b in range(len(ypbatch)):
                image = ypbatch[b]
                path = paths[b]
                _, name = os.path.split(path)
                output_path = os.path.join(output_dir, name)
                for i in range(image.shape[2]):
                    # im = Image.fromarray(image[:, :, i].astype('float32'))
                    # im.save(output_path+'_'+output_channels[i]+'_output.tif')
                    misc.imsave(output_path + '_' + output_channels[i] + '_output.tif',
                                image[:, :, i].astype('float32'))
            api.showProgress(1.0 * count / totalsize)
            api.showStatus('making predictions: {}/{}'.format(count, totalsize))

    async def train(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before training.')
            return
        opt = self._opt
        sources = GenericTransformedImages(opt)
        epochs = my.config.epochs
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Training", w=25, h=10,
                                           data={"display_mode": "all", 'metrics': ['mse', 'dssim_l1'],
                                                 'callbacks': ['onStep']})
        updateUI = UpdateUI(epochs, self.dash, make_generator(sources['valid'], batch_size=1), opt)
        opt.batch_size = my.config.batchsize
        abort.clear()
        tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, my.config.name + 'logs'), histogram_freq=0,
                                  batch_size=32, write_graph=True, write_grads=False, write_images=True)
        checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir, my.config.name + '__model__.hdf5'),
                                       verbose=1, save_best_only=True)
        self.model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                                 validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                                 validation_steps=4, steps_per_epoch=my.config.steps, epochs=epochs, verbose=2,
                                 callbacks=[updateUI, checkpointer, tensorboard])
        self.model.save(os.path.join(opt.checkpoints_dir, my.config.name + '__model__.hdf5'))

        model_config = {}
        model_config['input_size'] = opt.input_size
        model_config['input_channels'] = len(opt.input_channels)
        model_config['target_channels'] = len(opt.target_channels)

        with open(os.path.join(opt.work_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)

    async def train_run(self, my):
        # json_path = "datasets/home/anet_png/config.json"
        json_path = "datasets/home/example03/config.json"
        print("json_path:", json_path)

        with open(json_path, "r") as f:
            json_content =f.read()
        config_json = json.loads(json_content)
        print("config_json:", config_json)
        # await self.get_data_by_config(config=config_json)
        self.get_mask_by_json(config=config_json)

        self._opt = my_opt(config_json)
        self.initialize(self._opt)
        print("self._opt.work_dir:", self._opt.work_dir)
        print("self._opt.input_channels:", self._opt.input_channels)
        print("self._opt.target_channels:", self._opt.target_channels)
        print("self._opt.input_nc:", self._opt.input_nc)
        print("self._opt.target_nc:", self._opt.target_nc)

        config = my_config()
        print("config.name:", config.name)
        print("config.epochs:", config.epochs)
        print("config.steps:", config.steps)
        print("config.batchsize:", config.batchsize)

        # if no valid copy train as valid
        if not os.path.exists(os.path.join(self._opt.work_dir, "valid")):
            shutil.copytree(os.path.join(self._opt.work_dir, "train"), os.path.join(self._opt.work_dir, "valid"))

        await self.train_2(config)
        pass

    async def test_run(self, my):
        if self._initialized:
            samples = os.listdir(os.path.join(self._opt.work_dir, "test"))
        else:
            json_path = "datasets/home/example03/config.json"
            print("json_path:", json_path)

            with open(json_path, "r") as f:
                json_content =f.read()
            config_json = json.loads(json_content)
            print("config_json:", config_json)
            # await self.get_data_by_config(config=config_json)
            # self.get_mask_by_json(config=config_json)

            self._opt = my_opt(config_json)
            self._opt.load_from = "datasets/home/example03/__model__/__model__.hdf5"
            self.initialize(self._opt)

            print("self._opt.work_dir:", self._opt.work_dir)
            print("self._opt.input_channels:", self._opt.input_channels)
            print("self._opt.target_channels:", self._opt.target_channels)
            print("self._opt.input_nc:", self._opt.input_nc)
            print("self._opt.target_nc:", self._opt.target_nc)

            samples = os.listdir(os.path.join(self._opt.work_dir, "test"))
            self._initialized = True

        sample_path = {"samples": [os.path.join(self._opt.work_dir, "test", samples[0])]}
        await self.auto_test(samples=sample_path)
        # await self.auto_test_2()
        pass

    async def train_2(self, config):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before training.')
            return
        opt = self._opt

        sources = GenericTransformedImages(opt)
        epochs = config.epochs
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Training", w=25, h=10,
                                           data={"display_mode": "all", 'metrics': ['mse', 'dssim_l1'],
                                                 'callbacks': ['onStep']})
        updateUI = UpdateUI(epochs, self.dash, make_generator(sources['valid'], batch_size=1), opt)
        # updateUI = []
        opt.batch_size = config.batchsize
        abort.clear()
        tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, config.name + 'logs'), histogram_freq=0,
                                  batch_size=32, write_graph=True, write_grads=False, write_images=True)
        checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir, config.name + '__model__.hdf5'),
                                       verbose=1, save_best_only=True)
        self.model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                                 validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                                 validation_steps=4, steps_per_epoch=config.steps, epochs=epochs, verbose=2,
                                 callbacks=[updateUI, checkpointer, tensorboard])
        self.model.save(os.path.join(opt.checkpoints_dir, config.name + '__model__.hdf5'))

        model_config = {}
        model_config['input_size'] = opt.input_size
        model_config['input_channels'] = len(opt.input_channels)
        model_config['target_channels'] = len(opt.target_channels)

        with open(os.path.join(opt.work_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)

    async def auto_train(self, configPath):
        print("configPath:", configPath)
        json_path = "datasets" + configPath["configPath"]
        json_content = await self.readFile(configPath["configPath"])
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))
        with open(json_path, "w") as f:
            f.write(json_content)
            print("config.json save to path :", json_path)

        config_json = json.loads(json_content)
        print("config_json:", config_json)

        await self.get_data_by_config(config=config_json)
        self.get_mask_by_json(config=config_json)

        self._opt = my_opt(config_json)
        self.initialize(self._opt)
        print("self._opt.work_dir:", self._opt.work_dir)
        print("self._opt.input_channels:", self._opt.input_channels)
        print("self._opt.target_channels:", self._opt.target_channels)
        print("self._opt.input_nc:", self._opt.input_nc)
        print("self._opt.target_nc:", self._opt.target_nc)

        config = my_config()
        print("config.name:", config.name)
        print("config.epochs:", config.epochs)
        print("config.steps:", config.steps)
        print("config.batchsize:", config.batchsize)

        if not os.path.exists(os.path.join(self._opt.work_dir, "valid")):
            # copy train dir as valid dir
            shutil.copytree(os.path.join(self._opt.work_dir, "train"), os.path.join(self._opt.work_dir, "valid"))
#         network_config = {
#   "api_version": "0.1.3",
#   "channel_config": {
#     "002": {
#       "filter": "c002",
#       "name": "c002"
#     },
#     "007": {
#       "filter": "c007",
#       "name": "c007"
#     }
#   },
#   "annotation_types": {
#     "cell": {
#       "label": "cell",
#       "color": "#ff0000",
#       "line_width": 4,
#       "type": "Polygon"
#     },
#     "44444": {
#       "label": "44444",
#       "color": "#009688",
#       "line_width": 4,
#       "type": "LineString"
#     }
#   },
#   "network_types": [
#     {
#       "type": "unet"
#     }
#   ],
#   "post_processing_types": [
#     {
#       "name": "withseed",
#       "type": "withseed"
#     },
#     {
#       "name": "seedless",
#       "type": "seedless",
#       "options": [
#         {
#           "type": "string",
#           "name": "seed"
#         }
#       ]
#     }
#   ],
#   "loss_types": [
#     {
#       "type": "mse"
#     }
#   ]
# }
        network_config = {
          "api_version": "0.1.3",
          "channel_config": config_json.get("channel_config"),
          "annotation_types": config_json.get("annotation_types"),
          "network_types": [{"type": "unet"}],
          "post_processing_types": [{"name": "withseed", "type": "withseed"},
                                    {"name": "seedless", "type": "seedless",
                                     "options": [{"type": "string","name": "seed"}]}],
          "loss_types": [{"type": "mse"},
                         {"type": "cross entropy"}]
        }
        # await api.createWindow(type="NetworkConfig",
        #                        data={"finish_callback": "finish_config_callback",
        #                              "config": network_config})
        await self.train_2(config)

    async def finish_config_callback(self, config):
        await print("config:", config)
        # inputs = config.inputs
        # targets = config.targets
        # post_processing = config.post_processing
        # await self.train_2(config)

    def cus_make_test_generator(self, source, sample_path, batch_size=1):
        x, path = [], []
        count = 0
        for d in source:
            print("d['path']:", d['path'])
            print("sample_path:", sample_path)
            if d['path'] == sample_path:
                print("d['path']:", d['path'])
                x.append(d['A'])
                path.append(d['path'])
                if len(x) >= batch_size:
                    x = np.stack(x, axis=0)
                    m, s = x.mean(), x.std() + 0.0001
                    x = (x - m) / s
                    yield x, path
                    x, path = [], []
                    count += 1
        return

    async def auto_test(self, samples):
        sample_path = samples["samples"][0]
        print("start run GenericTransformedImages ...")
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before testing.')
            return
        sources = GenericTransformedImages(self._opt)
        batch_size = 1
        source = sources['test']
        count = 0

        # print("start run cus_make_test_generator ...")
        gen = self.cus_make_test_generator(source, sample_path)
        # gen = make_test_generator(source)
        api.showStatus('making predictions.')
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10,
                                           data={"display_mode": "all"})

        input_channels = [ch[0] for ch in self._opt.input_channels]
        output_channels = [ch[0] + '_output'  for ch in self._opt.target_channels]
        label = 'Sample '
        titles = [input_channels, output_channels]
        print("titles:", titles)

        xbatch, paths = next(gen)
        print("start run predict ...")
        ypbatch = self.model.predict(xbatch, batch_size=batch_size)
        tensor_list = [xbatch, ypbatch]
        plot_tensors(self.dash, tensor_list, label, titles)
        count += batch_size
        for b in range(len(ypbatch)):
            image = ypbatch[b]
            path = paths[b]
            _, name = os.path.split(path)
            # output_path = os.path.join(sample_path, name)
            for i in range(image.shape[2]):
                misc.imsave(os.path.join(sample_path, output_channels[i] + '.png'),
                            image[:, :, i].astype('float32'))
        api.showProgress(1.0 * count / totalsize)
        api.showStatus('making predictions: {}/{}'.format(count, totalsize))
        # mask_file = os.path.join(sample_path, output_channels[0] + '.png')
        # save_name = os.path.join(sample_path, 'annotation_MASK.json')
        annotation_string = json.dumps(masks_to_annotation(sample_path))
        fs_path = sample_path.replace("datasets", "")
        fs_path = "/tmp/prediction.json"
        print("save prediction.json to browser fs_path:", fs_path)
        try:
            await self.writeFile(fs_path, annotation_string)
            # return fs_path
        except:
            print("write data to file: {} error.".format(sample_path.replace("datasets", "")))

        # file_content = await self.readFile(fs_path)
        # # print("file_content:", file_content)
        # with open(fs_path, "w") as f:
        #     f.write(file_content)

        # return annotation_string

    async def auto_test_2(self):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before testing.')
            return
        sources = GenericTransformedImages(self._opt)
        batch_size = 1
        source = sources['test']
        count = 0

        gen = make_test_generator(source)
        api.showStatus('making predictions.')
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10,
                                           data={"display_mode": "all"})

        input_channels = [ch[0] for ch in self._opt.input_channels]
        output_channels = [ch[0] + '_output' for ch in self._opt.target_channels]

        titles = [input_channels, output_channels]
        print("titles:", titles)
        for j in range(int(totalsize/batch_size+0.5)):
            xbatch, paths = next(gen)
            ypbatch = self.model.predict(xbatch, batch_size=batch_size)
            tensor_list = [xbatch, ypbatch]
            label = 'Sample '+ str(j)
            titles = [input_channels, output_channels]
            plot_tensors(self.dash, tensor_list, label, titles)
            count +=batch_size
            for b in range(len(ypbatch)):
                image = ypbatch[b]
                path = paths[b]
                print("path:", path)

                for i in range(image.shape[2]):
                    print("output_channels[i]:", output_channels[i])
                    misc.imsave(os.path.join(path, output_channels[i] + '.png'),
                                image[:, :, i].astype('float32'))
                annotation_string = json.dumps(masks_to_annotation(path))
                # return annotation_string

            api.showProgress(1.0*count/totalsize)
            api.showStatus('making predictions: {}/{}'.format(count, totalsize))

    async def get_data_by_config(self, config):
        if config["root_folder"].startswith("/"):
            work_dir = "datasets" + config["root_folder"]
        else:
            work_dir = config["root_folder"]

        samples = config["samples"]
        for sample in samples:
            saved_dir = os.path.join(work_dir, sample["group"], sample["name"])
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)

            # save img file
            sample_data = sample["data"]
            for key in sample_data.keys():
                file_fs_path = os.path.join(config["root_folder"], sample["group"], sample["name"], sample_data[key]["file_name"])
                file_path = os.path.join(work_dir, sample["group"], sample["name"], sample_data[key]["file_name"])
                try:
                    file_content = await self.readFile(file_fs_path)
                    if file_path.endswith(".base64"):
                        file_path = file_path.replace(".base64", "")
                        with open(file_path, "wb") as f:
                            f.write(bytes(base64.b64decode(file_content.replace("data:image/png;base64,", ""))))
                    else:
                        with open(file_path, "w") as f:
                            f.write(file_content)
                    # print("file_fs_path:", file_fs_path)
                    # print("file_path:", file_path)
                except:
                    print("warnming: can not get the file :", file_fs_path)

    def get_mask_by_json(self, config):
        if config["root_folder"].startswith("/"):
            work_dir = "datasets" + config["root_folder"]
        else:
            work_dir = config["root_folder"]

        samples = config["samples"]
        anno_path_list = []
        for sample in samples:
            try:
                anno_name = sample.get("data").get("annotation.json").get("file_name")
            except:
                print("can not get annotation json file form sample:", sample["name"])
                print("using default annotation.json file.")
                anno_name = "annotation.json"
                # annotation path
            anno_path = os.path.join(work_dir, sample["group"], sample["name"], anno_name)
            print("anno_path:", anno_path)
            anno_path_list.append(anno_path)
        # size_file  = os.path.join(work_dir, samples[0]["group"], samples[0]["name"], samples[0]["data"][0]["name"].replace(".base64", ""))
        # mask_img = io.imread(size_file)
        # print("mask_img.shape:", mask_img.shape)
        gen_mask_from_geojson(files_proc=anno_path_list, img_size=(624, 924))
        return True
        pass

    def readFile(self, path):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        def cb(err, data=None):
            if err:
                fut.set_exception(Exception(err))
                return
            fut.set_result(data)

        api.fs.readFile(path, 'utf8', cb)
        return fut

    def fs_readfile(self, path):
        saved_dir = "datasets" + os.path.dirname(path)
        saved_path = "datasets" +  path
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def cb(err, data=None):
            if err:
                print(err)
                fut.set_exception(Exception(err))
                return
            png_saved_path = saved_path
            if png_saved_path.endswith(".png.base64"):
                png_saved_path = png_saved_path.replace(".png.base64", ".png")
                with open(png_saved_path, "wb") as f:
                    f.write(bytes(base64.b64decode(data.replace("data:image/png;base64,", ""))))
            else:
                with open(png_saved_path, "w") as f:
                    f.write(data)

            fut.set_result(png_saved_path)

        api.fs.readFile(path, 'utf8', cb)
        return fut

    def writeFile(self, path, file_data):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        def read(err, data=None):
            if err:
                print(err)
                return

            fut.set_result(path)

        api.fs.writeFile(path, file_data, read)
        return fut

    async def freeze_model(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before loading weights.')
            return
        opt = self._opt
        tf.identity(tf.get_default_graph().get_tensor_by_name(self.model.outputs[0].op.name + ':0'), name="unet_output")
        frozen_graph = freeze_session(K.get_session(),
                                      output_names=['unet_output'])

        config = json.loads(await api.getAttachment('model_config_template'))

        config['label'] = 'Unet_{}x{}_{}_{}'.format(opt.input_size, opt.input_size, len(opt.input_channels),
                                                    len(opt.target_channels))
        config['model_name'] = config['label']

        config['inputs'][0]['key'] = 'unet_input'
        config['inputs'][0]['channels'] = [ch[0] for ch in opt.input_channels]
        config['inputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.input_channels)]
        config['inputs'][0]['size'] = opt.input_size

        config['outputs'][0]['key'] = 'unet_output'
        config['outputs'][0]['channels'] = [ch[0] for ch in opt.target_channels]
        config['outputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.target_channels)]
        config['outputs'][0]['size'] = opt.input_size

        with open(os.path.join(opt.work_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        tf.train.write_graph(frozen_graph, opt.work_dir, "tensorflow_model.pb", as_text=False)

        api.alert('model has been exported as ' + os.path.abspath(os.path.join(opt.work_dir, "tensorflow_model.pb")))

api.export(ImJoyPlugin())

# if __name__ == '__main__':
#     test_run = ImJoyPlugin()
#     test_run.train_run("")

</script>