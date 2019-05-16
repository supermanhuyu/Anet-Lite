<docs lang="markdown">
# Anet-lite

A generic plugin for image-to-image translation with A-net.

## Usage

### Data preparation
Your data should be organized according to the following structure

```
 - train
    - sample1
      - channelA.png
      - channelB.png
      - channelC.png
    - sample2
      - channelA.png
      - channelB.png
      - channelC.png
    - ...
 - valid
    - sample20
      - channelA.png
      - channelB.png
      - channelC.png
    - ...
 - test
    - sample43
      - channelA.png
      - channelC.png
    - sample44
      - channelA.png
      - channelC.png
```
In the above folder structure, `train/valid/test` are three folders, `sample1`...`sample44` are sample folders, within the sample folder, it contains images with different channels.

For the naming, you have to use `train`/`valid`/`test` as the top level folder name, but the sample name can be choose freely.
You can also choose different channel names (e.g. `DAPI`, `C5`), but you need to make it consistent accross all the sample folder.

These channel names will be used as an identifier for the plugin to recognize them, you will be asked to specify them when you `set working directory`.

### Set working directory and parameters

If you have the samples pepared the next step is to `set working directory` to the root folder of your samples.

And you will be asked to specify the identifiers for your plugin.

The following parameter can be configured:
 * input identifiers and output identifiers: a string which specifies the naming pattern of each input and output channels, for example, the above folder structure contains `channelA.png`, `channelB.png` and `channelC.png`,
 if we want to use all the `channelA.png` and `channelC.png` as input channels, we can set `Cells=channelA*.png,DAPI=channelC*.png`,
 where before `=` is the name we given for each channel, and we used `,` to seperate channels. Similary for the input channels, we can specify `Mask=channelB*.png`.

Here we used a `*` here which means it can be replaced with any symbol, here there is no extra, it is designed for when there are multiple images belongs to the same channel.

## Optionally, loading previously trained models

## start the training

The training will relying on two folders named `train` and `valid`, you need to make sure you have them in your working directory.

You can specify a training name, which will be used a prefix for saving models and logs.
Choose the `epochs`, `step per epoch` and `batch size` you want to train.

Then you can start training by click on the plugin menu.

If the training is done, you will get your model located in the `working directory`, the folder will be called `__model__`.

You can load the file named `*__model__.h5` next time if you want to start another training with this one (aka warm start).

Or you can load the trained model for testing new images.

## Testing/Inferencing
Place your files with all the input channels in a folder `test` and you will be able to run prediction on them.

By clicking `test` in the plugin menu, you will start the prediction.

Once done, the result will be saved automatically into the testing folder.

</docs>

<config lang="json">
{
  "name": "Anet-Lite",
  "type": "native-python",
  "version": "0.3.6",
  "api_version": "0.1.3",
  "description": "A generic plugin for image-to-image translation with A-net.",
  "tags": ["CPU", "GPU", "Windows-CPU", "Window-GPU"],
  "ui": [],
  "inputs": null,
  "outputs": null,
  "icon": null,
  "env": {
    "CPU":["conda create -n anet-cpu python=3.6"],
    "GPU": ["conda create -n anet-gpu2 python=3.6"],
    "Windows-CPU": ["conda create -n anet-win-cpu python=3.6"],
    "Windows-GPU": ["conda create -n anet-win-gpu python=3.6"]
  },
  "requirements": {"CPU":["repo: https://github.com/oeway/Anet-Lite", "read_roi", "scikit-image", "geojson", "shapely", "descartes", "geojson", "palettable", "tensorflow==1.5", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "GPU": ["repo: https://github.com/oeway/Anet-Lite", "read_roi", "scikit-image", "geojson", "shapely", "descartes", "geojson", "palettable", "tensorflow-gpu==1.5", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "Windows-CPU":["repo: https://github.com/oeway/Anet-Lite", "read_roi", "scikit-image", "geojson", "shapely", "descartes", "geojson", "palettable", "tensorflow==1.2", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "Windows-GPU": ["repo: https://github.com/oeway/Anet-Lite", "read_roi", "scikit-image", "geojson", "shapely", "descartes", "geojson", "palettable", "tensorflow-gpu==1.2", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"]
   },
   "flags": [],
  "dependencies": ["oeway/ImJoy-Plugins:Im2Im-Dashboard",
                   "https://git.sg-ai.com/imjoy/ImJoy-Plugins/raw/master/plugins/ImageSelection.imjoy.html",
                   "https://git.sg-ai.com/imjoy/ImJoy-Plugins/raw/master/plugins/AnetConfig.imjoy.html"]
}
</config>

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
# os.chdir('..')

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
    def __init__(self, name="", epochs=10, batchsize=4, steps=10):
        self.name = name
        self.epochs = epochs
        self.steps = steps
        self.batchsize = batchsize
        self.config_win = None


class ImJoyPlugin():
    def __init__(self):
        self._initialized = False

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
        api.register(name="train_test", run=self.train_test, ui="train")
        api.register(name="predict_test", run=self.predict, ui="predict")
        pass

    async def run(self, my):
        self.Img_Select_window = await api.showDialog({
            "name": "ImageSelection",
            "type": "ImageSelection",
            "data": {
                "callback": self.train
            }
        })
        # await self.train("")
        print("Img_Select_window:", self.Img_Select_window)
        # configPath = {"configPath": os.path.join(datasets_dir, "config.json")}
        # await self.auto_train(configPath=configPath)

    async def train(self, config_json, Json_Status):
        self.Img_Select_window.close()
        # weight_path = await api.showFileDialog(root=os.getcwd(), type="directory")
        print("config_json:", config_json)
        json_path = os.path.join(config_json["root_folder"], "config.json")
        with open(json_path, "w") as f:
            f.write(json.dumps(config_json))
        # configPath = {"configPath": json_path}
        # await self.auto_train(configPath=configPath)

        self.config_json = config_json
        self.work_dir = config_json["root_folder"]
        print("self.work_dir:", self.work_dir)

        self.config_win = await api.showDialog({
            "name": 'AnetConfig',
            "type": 'AnetConfig',
            "w": 20, "h": 15,
            # "fullscreen": True,
            "data": {
                "configJson": self.train_config(),
                "callback": self.finish_config_callback
            }
        })

    async def predict(self, my=None, sample_path=None):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before testing.')
            return

        sources = GenericTransformedImages(self._opt)
        batch_size = 1
        source = sources['test']
        count = 0

        if sample_path is None:
            test_samples = os.listdir(os.path.join(self.work_dir, "test"))
            sample_path = os.path.join(self.work_dir, "test", test_samples[0])
        # print("start run cus_make_test_generator ...")
        gen = self.cus_make_test_generator(source, sample_path)
        # gen = make_test_generator(source)
        api.showStatus('making predictions.')
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10,
                                           data={"display_mode": "all"})

        input_channels = [ch[0] for ch in self._opt.input_channels]
        output_channels = [ch[0] + '_output' for ch in self._opt.target_channels]
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
        annotation_json = self.masks_to_annotation(sample_path, outputs=self.config_json.get("outputs"))
        print("save prediction.json to path:", sample_path)
        return annotation_json

    async def add_train_run(self, my):
        sample_path = "datasets/example/test/z018"
        await self.add_training_data(sample_path, local_anno=True)

    async def train_2(self, config):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before training.')
            return
        opt = self._opt

        sources = GenericTransformedImages(opt)
        epochs = config.epochs
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Training", w=20, h=15,
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

    def train_config(self):
        post_processing_types = ["withseed", "seedless"]
        loss_types = ["mse", "cross entropy"]
        target_types = ["channel", "annotation"]
        target_masks_type = ["filled", "distmap", "edge", "weighted"]
        network_types = ["Anet", "Unet"]
        channel_config = self.config_json.get("channel_config")
        annotation_types = self.config_json.get("annotation_types")

        default_values = None
        if os.path.exists(os.path.join(self.work_dir, "anet-config.json")):
            with open(os.path.join(self.work_dir, "anet-config.json"), "r") as f:
                default_values = json.load(f)

        outputs_config = []
        if default_values is None:
            default_values = {
                "base_filter": "16",
                "inputs": list(channel_config.keys()),
                "outputs": outputs_config,
                "network": {"type": "Anet"}
            }

        for label in annotation_types.keys():
            outputs_config.append({
                "loss": {
                    "type": loss_types[0]
                },
                "name": label + "_" + target_masks_type[0],
                "postProcessing": {
                    "name": "seedless",
                    "type": "seedless",
                    "seed": ""
                },
                "target": {
                    "name": label + "_" + target_masks_type[0],
                    "type": "annotation"
                }})
            outputs_config.append({
                "loss": {
                    "type": loss_types[0]
                },
                "name": label + "_" + target_masks_type[1],
                "postProcessing": {
                    "name": "withseed",
                    "type": "withseed",
                    "seed": label + "_" + target_masks_type[0]
                },
                "target": {
                    "name": label + "_" + target_masks_type[1],
                    "type": "annotation"
                }})

        network_config = {
            "api_version": "0.1.3",
            "channel_config": channel_config,
            "annotation_types": annotation_types,
            "post_processing_types": [{"name": pp_type, "type": pp_type} for pp_type in post_processing_types],
            "loss_types": [{"type": loss} for loss in loss_types],
            "target_types": [{"type": t_type} for t_type in target_types],
            "target_masks": [{"type": t_mask_type} for t_mask_type in target_masks_type],
            "network_types": [{"type": net} for net in network_types],
            "default": default_values}

        print("network_config:", network_config)
        return network_config

    async def finish_config_callback(self, callback_config):
        print("callback_config:", callback_config)
        if self.config_win is not None:
            self.config_win.close()
        api.showMessage('network configured.')
        with open(os.path.join(self.work_dir, "anet-config.json"), "w") as f:
            f.write(json.dumps(callback_config))
        self.config_json.update(callback_config)

        # await self.get_data_by_config(config=self.config_json)
        api.showStatus("generating mask from the annotation file ...")
        self.get_mask_by_json(config=self.config_json)
        api.showStatus("Masks generated, now start training...")
        self._opt = self.get_opt(self.config_json, work_dir=self.work_dir)
        self.initialize(self._opt)
        api.log("self._opt.work_dir:" + self._opt.work_dir)
        api.log("self._opt.input_channels:" + str(self._opt.input_channels))
        api.log("self._opt.target_channels:" + str(self._opt.target_channels))
        api.log("self._opt.input_nc:" + str(self._opt.input_nc))
        api.log("self._opt.target_nc:" + str(self._opt.target_nc))

        config = my_config()
        api.log("config.name:" + config.name)
        api.log("config.epochs:" + str(config.epochs))
        api.log("config.steps:" + str(config.steps))
        api.log("config.batchsize:" + str(config.batchsize))

        await self.train_2(config)

    async def train_test(self, my=None, json_path=None):
        print("os.getcwd():", os.getcwd())
        print("my:", my)
        # print("json_path:", json_path)
        if json_path is None:
            json_path = "datasets/data_bak/ex/example_anno/config.json"

        with open(json_path, "r") as f:
            config_json = json.loads(f.read())
        self.config_json = config_json
        self.work_dir = config_json["root_folder"]
        print("self.work_dir:", self.work_dir)

        self.config_win = await api.showDialog({
            "name": 'AnetConfig',
            "type": 'AnetConfig',
            "w": 20, "h": 15,
            # "fullscreen": True,
            "data": {
                "configJson": self.train_config(),
                "callback": self.finish_config_callback
            }
        })

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

    async def add_training_data(self, sample_path, local_anno=False):
        # get annotation.json
        file_fs_path = os.path.join(sample_path, "annotation.json")
        anno_path = file_fs_path
        if not local_anno:
            file_content = await self.readFile(file_fs_path)
            # anno_path = "datasets" + file_fs_path
            with open(anno_path, "w") as f:
                f.write(file_content)

        # generate mask
        if os.path.exists(anno_path):
            print("generate mask from file:", anno_path)
            self.gen_mask_from_geojson(files_proc=[anno_path])
        else:
            print("can not find annotation file:", anno_path)

        # mv to train dir
        shutil.move(
            os.path.dirname(anno_path),
            os.path.join(self._opt.work_dir, "train"))

        # update training generate
        self.sources = GenericTransformedImages(self._opt)
        pass

    async def get_data_by_config(self, config):
        samples = config["samples"]
        for sample in samples:
            saved_dir = os.path.join(self.work_dir, sample["group"], sample["name"])
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)

            # save img file
            sample_data = sample["data"]
            for key in sample_data.keys():
                file_fs_path = os.path.join(config["root_folder"], sample["group"], sample["name"],
                                            sample_data[key]["file_name"])
                file_path = os.path.join(self.work_dir, sample["group"], sample["name"], sample_data[key]["file_name"])
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
        samples = config["samples"]
        anno_path_list = []
        for sample in samples:
            if sample["group"] != "test":
                try:
                    anno_name = sample.get("data").get("annotation.json").get("file_name")
                except:
                    print("can not get annotation json file form sample:", sample["name"])
                    print("using default annotation.json file.")
                    anno_name = "annotation.json"
                    # annotation path
                anno_path = os.path.join(self.work_dir, sample["group"], sample["name"], anno_name)
                if os.path.exists(anno_path):
                    print("anno_path:", anno_path)
                    anno_path_list.append(anno_path)
                else:
                    print("can not find annotation file:", anno_path)
            else:
                print("skip generate mask from test group.")
        if len(anno_path_list) != 0:
            with open(anno_path_list[0]) as f:
                anno_json = json.loads(f.read())
                bbox = anno_json.get("bbox")
                self.gen_mask_from_geojson(files_proc=anno_path_list, img_size=(bbox[2], bbox[3]))
            # self.gen_mask_from_geojson(files_proc=anno_path_list)
        return True

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
        saved_path = "datasets" + path
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

    async def get_engine(self):
        print("api.ENGINE_URL:", api.ENGINE_URL)
        return api.ENGINE_URL

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

    def get_opt(self, config, work_dir):
        opt = Options().parse(['--work_dir={}'.format(work_dir)])
        opt.work_dir = work_dir
        opt.input_size = 256
        opt.base_filter = int(config.get("base_filter"))
        opt.input_channels = []
        opt.target_channels = []
        # opt.load_from = None
        # opt.checkpoints_dir = opt.work_dir + "/__model__"

        inputs = config.get("inputs")
        outputs = config.get("outputs")
        # network = config.get("network")

        opt.channel = config.get("channel_config")
        for key in [input_key for input_key in opt.channel.keys() if input_key in inputs]:
            opt.input_channels.append(
                (opt.channel[key]["name"], {'filter': "*" + opt.channel[key]["filter"] + "*", 'loader': ImageLoader()},))

        for out in outputs:
            print("add target_channel:", out.get("name"))
            opt.target_channels.append((out.get("name"), {'filter': "*" + out.get("name") + "*", 'loader': ImageLoader()},))

        opt.input_nc = len(opt.input_channels)
        opt.target_nc = len(opt.target_channels)
        return opt

    def gen_mask_from_geojson(self, files_proc, img_size=None, infer=False):
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

        # %% Loop over all files
        count = len(files_proc)
        for i, file_proc in enumerate(files_proc):
            print('PROCESSING FILE:')
            print(file_proc)
            dir_name, file_name = os.path.split(file_proc)
            api.showStatus('generating masks for: ' + dir_name.split('/')[-1])
            api.showProgress(i / count * 100)
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
                # print("annot_type: ", annot_type)
                masks_to_create[annot_type] = masks_to_create_value

                # Filter the annotations by label
                annot_dict = {k: annot_dict_all[k] for k in annot_dict_all.keys() if
                              annot_dict_all[k]['properties']['label'] == annot_type}
                # print("len(annot_dict):", len(annot_dict))
                # print("annot_dict.keys():", annot_dict.keys())

                # Create masks

                # Binary - is always necessary to creat other masks
                print(' .... creating binary masks .....')
                binaryMasks = annotationUtils.BinaryMaskGenerator(image_size=image_size, erose_size=5, obj_size_rem=500,
                                                                  save_indiv=True)
                mask_dict = binaryMasks.generate(annot_dict)

                # Save binary masks FILLED if specified
                if 'filled' in masks_to_create[annot_type]:
                    if infer:
                        file_name_save = os.path.join(drive, path, annot_type + '_filled_output.png')
                    else:
                        file_name_save = os.path.join(drive, path, annot_type + '_filled.png')
                    masks.save(mask_dict, 'fill', file_name_save)

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
                    mask_dict = distMapMasks.generate(annot_dict, mask_dict)

                    # Save
                    if infer:
                        file_name_save = os.path.join(drive, path, annot_type + '_distmap_output.png')
                    else:
                        file_name_save = os.path.join(drive, path, annot_type + '_distmap.png')
                    masks.save(mask_dict, 'distance_map', file_name_save)

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

    def masks_to_annotation(self, outputs_dir, outputs=None):
        # %% Process one folder and save as one json file allowing multiple annotation types
        simplify_tol = 0  # Tolerance for polygon simplification with shapely (0 to not simplify)

        if os.path.exists(outputs_dir):
            print(f'Analyzing folder:{outputs_dir}')
            features = []  # For geojson
            image_size = None
            for out in outputs:
                file_full = os.path.join(outputs_dir, out.get("name") + "_output.png")
                if os.path.exists(file_full):
                    print("get output file path:", file_full)
                    mask_img = io.imread(file_full)
                    print("mask_img.shape:", mask_img.shape)
                    mask = measure.label(mask_img)
                    post_mask = DRFNStoolbox.seedless_segment(mask, 15, p_thresh=0.5)
                    img = Image.fromarray(post_mask.astype('uint8'))
                    img.save(os.path.join(outputs_dir, out.get("name") + '_noSeeds_OBJECTS.png'))
                    if out.get("postProcessing").get("type") == "withseed":
                        seed_mask_file = os.path.join(outputs_dir, out.get("postProcessing").get("seed") + "_output.png")
                        if os.path.exists(seed_mask_file):
                            print("get seed mask file :", seed_mask_file)
                            seed_mask_img = io.imread(seed_mask_file)
                            print("seed_mask_img.shape:", seed_mask_img.shape)
                            seed_mask = measure.label(seed_mask_img)
                            post_mask = DRFNStoolbox.segment_with_seed(seed_mask, post_mask, 15, p_thresh=0.5)
                            img = Image.fromarray(post_mask.astype('uint8'))
                            img.save(os.path.join(outputs_dir, out.get("name") + '_wSeeds_OBJECTS.png'))
                        else:
                            print("warming: seed mask file not exist:", seed_mask_file)

                    # Here summarizing the geojson should occur
                    image_size = mask_img.shape  # This might cause problems if any kind of binning was performed

                    # Get label from file name
                    label = out.get("name").split('_', 1)[0]
                    print("label:", label)
                    # print(mask_img[0:1, :100])

                    # Call function to transform segmentation masks into (geojson) polygons
                    feature, contours = segmentationUtils.masks_to_polygon(post_mask,
                                                                           label=label,
                                                                           simplify_tol=simplify_tol)
                    # save_name=file_full.replace(".png", ".json"))
                    features = features + feature
                else:
                    print("warming: output file not exist:", file_full)
            feature_collection = FeatureCollection(features, bbox=[0, 0.0, image_size[0], image_size[1]])

            # Save to json file
            save_name_json = os.path.join(outputs_dir, 'prediction.json')
            with open(save_name_json, 'w') as f:
                dump(feature_collection, f)
                f.close()
            return feature_collection

api.export(ImJoyPlugin())
</script>
