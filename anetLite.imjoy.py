import os
import sys
import requests
import zipfile
import random
import string
import asyncio
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.models import load_model
import base64
from io import BytesIO
import tensorflow as tf
import json
from keras import backend as K

os.chdir('Anet-Lite')
from anet.options import Options
from anet.data.examples import GenericTransformedImages
from anet.data.examples import TransformedTubulin001
from anet.data.file_loader import ImageLoader
from anet.data.utils import make_generator, make_test_generator
from anet.networks import UnetGenerator, get_dssim_l1_loss
from anet.utils import export_model_to_js
# import importlib
# importlib.reload(UnetGenerator)


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
        self.input_channels = []
        self.target_channels = []
        for i in opt.input_channels:
            self.input_channels.append(i[0])
        for i in opt.target_channels:
            self.target_channels.append(i[0])

    def on_batch_end(self, batch, logs):
        self.logs = logs
        api.showStatus('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' ' + str(logs))
        sys.stdout.flush()
        self.dash.updateCallback('onStep', self.step, {'mse': np.asscalar(logs['mean_squared_error']), 'dssim_l1': np.asscalar(logs['DSSIM_L1'])})
        self.step += 1

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        self.logs = logs
        api.showProgress(self.epoch/self.total_epoch*100)
        api.showStatus('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' '+ str(logs))
        xbatch, ybatch = next(self.gen)
        ypbatch = self.model.predict(xbatch, batch_size=1)
        tensor_list = [ypbatch, xbatch, ybatch]
        label = 'Step '+ str(self.step)
        titles = [["output"], self.input_channels, self.target_channels]
        plot_tensors(self.dash, tensor_list, label, titles)


class ImJoyPlugin():
    def __init__(self):
        self._initialized = False
    def download_data(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before downloading.')
            return
        work_dir = self._opt.work_dir
        target_dir = os.path.normpath(os.path.join(work_dir, '../example_data_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        url = 'https://www.dropbox.com/s/02w4he65f2cnf1t/EM_membrane_dataset.zip?dl=1'
        api.showStatus('downloading...')
        r = requests.get(url, allow_redirects=True)
        name_zip = os.path.join(work_dir,'EM_membrane_dataset.zip')
        open(name_zip, 'wb').write(r.content)
        api.showStatus('unzipping...')
        with zipfile.ZipFile(name_zip, 'r') as f:
            f.extractall(target_dir)
        # os.remove(name_zip)
        api.showStatus('example data saved to ' + os.path.abspath(target_dir))
        self._opt.work_dir = target_dir
    def initialize(self, opt):
        # opt.work_dir = '{}/unet_data/train'.format(os.getcwd())
        self.model = UnetGenerator(input_size=opt.input_size, input_channels=opt.input_nc, target_channels=opt.target_nc, base_filter=opt.base_filter)
        if opt.load_from is not None:
            print('loading weights from: ' + opt.load_from)
            self.model.load_weights(opt.load_from)
        else:
            model_path = os.path.join(opt.checkpoints_dir, '__model__.hdf5')
            if os.path.exists(model_path):
                print('loading weights from: ' + model_path)
                self.model.load_weights(model_path)
        DSSIM_L1 = get_dssim_l1_loss()
        self.model.compile(optimizer='adam',
                    loss=DSSIM_L1,
                    metrics=['mse', DSSIM_L1])
        self._initialized = True
        api.showStatus("A-net lite successfully initialized.")
    async def setup(self):
        #api.register(name="set working directory", run=self.set_work_dir, ui="set working directory for loading data and saving trained models")
        api.register(name="get example dataset", run=self.download_data, ui="download example data set to your workspace")
        api.register(name="load trained weights", run=self.load_model_weights, ui="load a trained weights for the model")
        api.register(name="train", run=self.train, ui="name:{id:'name', type:'string', placeholder:''}<br>"\
                "epochs:{id:'epochs', type:'number', min:1, placeholder:100}<br>"\
                "steps per epoch:{id:'steps', type:'number', min:1, placeholder:10}<br>"\
                "batch size:{id:'batchsize', type:'number', min:1, placeholder:4}<br>"
                )
        api.register(name="freeze and export model", run=self.freeze_model, ui="freeze and export the graph as pb file")
        api.register(name="test", run=self.test, ui="number of images:{id:'num', type:'number', min:1, placeholder:10}<br>")
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
                api.showStatus('weights loaded from '+ weight_path)
        except:
            pass
    def export(self, my):
        opt = self._opt
        export_model_to_js(self.model, opt.work_dir+'/__js_model__')
    async def run(self, my):
        work_dir = await api.showFileDialog(title="Please select a working directory (with data)")
        api.setConfig('work_dir', work_dir)
        config = my.config
        print(config)
        opt = Options().parse(['--work_dir={}'.format(work_dir)])
        opt.work_dir = work_dir
        opt.input_size = config.input_size
        opt.base_filter = config.base_filter_num
        opt.input_channels = []
        for ch in config['input_ids'].split(','):
            name, filter = ch.split('=')
            opt.input_channels.append((name, {'filter':filter, 'loader':ImageLoader()}, ))
        opt.target_channels = []
        for ch in config['target_ids'].split(','):
            name, filter = ch.split('=')
            opt.target_channels.append((name, {'filter':filter, 'loader':ImageLoader()}, ))
        opt.input_nc = len(opt.input_channels)
        opt.target_nc = len(opt.target_channels)
        print(opt.input_channels, opt.target_channels)
        # opt.input_channels = [('cell', {'filter':'cells*.png', 'loader':ImageLoader()})]
        # opt.target_channels = [('mask', {'filter':'mask_edge*.png', 'loader':ImageLoader()})]
        self._opt = opt
        self.initialize(opt)
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
        output_channels = [c[0] for c in self._opt.target_channels]
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10, data={"display_mode": "all"})
        for i in range(int(totalsize/batch_size+0.5)):
            xbatch, paths = next(gen)
            ypbatch = self.model.predict(xbatch, batch_size=batch_size)
            tensor_list = [ypbatch, xbatch]
            label = 'Sample '+ str(i)
            titles = ["output", 'input']
            plot_tensors(self.dash, tensor_list, label, titles)
            count +=batch_size
            for b in range(len(ypbatch)):
                image = ypbatch[b]
                path = paths[b]
                _, name = os.path.split(path)
                output_path = os.path.join(output_dir, name)
                for i in range(image.shape[2]):
                    im = Image.fromarray(image[:, :, i].astype('float32'))
                    im.save(output_path+'_'+output_channels[i]+'_output.tif')
            api.showProgress(1.0*count/totalsize)
            api.showStatus('making predictions: {}/{}'.format(count, totalsize))
    async def train(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before training.')
            return
        opt = self._opt
        sources = GenericTransformedImages(opt)
        epochs =  my.config.epochs
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Training", w=25, h=10, data={"display_mode": "all", 'metrics': ['mse', 'dssim_l1'], 'callbacks': ['onStep']})
        updateUI = UpdateUI(epochs, self.dash, make_generator(sources['valid'], batch_size=1), opt)
        opt.batch_size = my.config.batchsize
        tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, my.config.name + 'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True)
        checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir,  my.config.name + '__model__.hdf5'), verbose=1, save_best_only=True)
        self.model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                            validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                            validation_steps=4, steps_per_epoch=my.config.steps, epochs=epochs, verbose=2, callbacks=[updateUI, checkpointer, tensorboard])
        self.model.save(os.path.join(opt.checkpoints_dir,  my.config.name + '__model__.hdf5'))

    async def freeze_model(self, my):
        if not self._initialized:
            api.alert('Please click `Anet-Lite` before loading weights.')
            return
        opt = self._opt
        tf.identity(tf.get_default_graph().get_tensor_by_name(self.model.outputs[0].op.name+':0'), name="unet_output")
        frozen_graph = freeze_session(K.get_session(),
                                    output_names=['unet_output'])

        config = json.loads(await api.getAttachment('model_config_template'))

        config['label'] = 'Unet_{}x{}_{}_{}'.format(opt.input_size, opt.input_size, len(opt.input_channels), len(opt.target_channels))
        config['model_name'] = config['label']

        config['inputs'][0]['key'] = 'unet_input'
        config['inputs'][0]['channels'] = [ ch[0] for ch in opt.input_channels]
        config['inputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.input_channels)]
        config['inputs'][0]['size'] = opt.input_size

        config['outputs'][0]['key'] = 'unet_output'
        config['outputs'][0]['channels'] = [ ch[0] for ch in opt.target_channels]
        config['outputs'][0]['shape'] = [1, opt.input_size, opt.input_size, len(opt.target_channels)]
        config['outputs'][0]['size'] = opt.input_size

        with open(os.path.join( opt.work_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

        tf.train.write_graph(frozen_graph, opt.work_dir, "tensorflow_model.pb", as_text=False)

        api.alert('model has been exported as ' + os.path.abspath(os.path.join(opt.work_dir, "tensorflow_model.pb")))

api.export(ImJoyPlugin())