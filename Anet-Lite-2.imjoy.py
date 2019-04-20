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
from skimage import io

os.chdir('Anet-Lite')
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.examples import TransformedTubulin001
from anet.data.file_loader import ImageLoader
from anet.data.utils import make_generator, make_test_generator
from anet.networks import UnetGenerator, get_dssim_l1_loss
from anet.utils import export_model_to_js
from generate_mask_geojson import generate_mask

# import importlib
# importlib.reload(UnetGenerator)

abort = threading.Event()


def plot_tensors(dash, tensor_list, label, titles):
    image_list = [tensor.reshape(tensor.shape[-3], tensor.shape[-2], -1) for tensor in tensor_list]
    displays = {}

    def stop():
        api.alert('stopped')
        abort.set()

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

# config_json = {
#     "samples": [{
#         "name":  "z010",
#         "annotation": "/tmp/example/train/z010/annotation.json",
#         "data": ["/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c002.png.base64",
#                  "/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c004.png.base64",
#                  "/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c007.png.base64"]
#     },{
#         "name":  "z010",
#         "annotation": "/tmp/example/train/z010/annotation.json",
#         "data": ["/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c002.png.base64",
#                  "/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c004.png.base64",
#                  "/tmp/example/train/z010/3500001066_100X_20170712_2r-Scene-6-P35-F05_z010_c007.png.base64"]
#     }],
#     "channel": {
#         "A": "c002",
#         "B": "c004",
#         "C": "c007",
#     }
# }

class my_config():
    def __init__(self, name="", epochs=100, batchsize=30, steps=2):
        self.name = name
        self.epochs = epochs
        self.steps = steps
        self.batchsize = batchsize

def my_opt(config, mask_config):
    work_dir = os.path.join("datasets", config["samples"][0]["data"][0].split("/")[2])
    opt = Options().parse(['--work_dir={}'.format(work_dir)])
    # self.work_dir = os.path.join("datasets", config["samples"][0]["data"][0].split("/")[2])
    opt.work_dir = work_dir
    opt.input_size = 256
    opt.base_filter = 16
    opt.input_channels = []
    opt.target_channels = []
    # opt.load_from = None
    # opt.checkpoints_dir = opt.work_dir + "/__model__"

    opt.channel = config["channel"]
    for name, filter_c in opt.channel.items():
        opt.input_channels.append((name, {'filter': "*"+filter_c+"*", 'loader': ImageLoader()},))

    opt.channel = mask_config["channel"]
    for name, filter_c in opt.channel.items():
        opt.target_channels.append((name, {'filter': filter_c, 'loader': ImageLoader()},))

    opt.input_nc = len(opt.input_channels)
    opt.target_nc = len(opt.target_channels)
    return opt


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
        # opt.work_dir = '{}/unet_data/train'.format(os.getcwd())
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
        # await self.fs_readfile(configPath["configPath"])
        json_path = configPath["configPath"].replace("/tmp", "datasets")
        json_content = await self.readFile(configPath["configPath"])
        with open(json_path, "w") as f:
            f.write(json_content)

        config_mask = {
            "channel": {
                "distMap": "*_distMap.png",
                "fill": "*_fill.png"
            }}
        config_json = json.loads(json_content)
        print("config_json:", config_json)
        await self.get_data_by_config(config=config_json)
        self.get_mask_by_json(config=config_json)

        self._opt = my_opt(config_json, config_mask)
        self.initialize(self._opt)
        # print("self._opt.work_dir:", self._opt.work_dir)
        # print("self._opt.input_channels:", self._opt.input_channels)
        # print("self._opt.target_channels:", self._opt.target_channels)

        config = my_config()
        # print("config.name:", config.name)
        # print("config.epochs:", config.epochs)
        # print("config.steps:", config.steps)
        # print("config.batchsize:", config.batchsize)

        if not os.path.exists(os.path.join(self._opt.work_dir, "valid")):
            # copy train dir as valid dir
            shutil.copytree(os.path.join(self._opt.work_dir, "train"), os.path.join(self._opt.work_dir, "valid"))
        await self.train_2(config)

    async def get_data_by_config(self, config):
        samples = config["samples"]
        for sample in samples:

            # save annotation file
            sample_annotation = sample["annotation"]
            saved_dir = os.path.dirname(sample_annotation).replace("/tmp", "datasets")
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            saved_path = sample_annotation.replace("/tmp", "datasets")
            anno_content = await self.readFile(sample_annotation)
            with open(saved_path, "w") as f:
                f.write(anno_content)

            # save img file
            sample_data = sample["data"]
            for file in sample_data:
                file_content = await self.readFile(file)
                file_path = file.replace("/tmp", "datasets")
                if file_path.endswith(".png.base64"):
                    file_path = file_path.replace(".png.base64", ".png")
                    with open(file_path, "wb") as f:
                        f.write(bytes(base64.b64decode(file_content.replace("data:image/png;base64,", ""))))
                else:
                    with open(file_path, "w") as f:
                        f.write(file_content)

    def get_mask_by_json(self, config):
        samples = config["samples"]
        mask = io.imread(samples[0]["data"][0].replace(".png.base64", ".png"))
        print("mask.shape:", mask.shape)
        for sample in samples:
            sample_annotation = sample["annotation"].replace("/tmp", "datasets")
            print(sample_annotation)
            generate_mask(files_proc=[sample_annotation], image_size=(mask.shape[0], mask.shape[1]))
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
        saved_dir = os.path.dirname(path).replace("/tmp", "datasets")
        saved_path = path.replace("/tmp", "datasets")

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

