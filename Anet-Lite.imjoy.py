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
  "version": "0.1.4",
  "api_version": "0.1.5",
  "description": "A generic plugin for image-to-image translation with A-net.",
  "tags": ["CPU", "GPU", "Windows-CPU", "Window-GPU"],
  "ui": ["Generic Image-to-Image Translation",
        "input identifiers: {id:'input_ids', type: 'string', placeholder: 'EM=EM*.png'}",
        "target identifiers: {id:'target_ids', type: 'string', placeholder: 'Mask=Mask*.png'}",
        "base filter num: {id:'base_filter_num', type: 'number', min: 8, placeholder: 16}",
        "input size: {id:'input_size', type: 'number', min: 256, placeholder: 256}"
    ],
  "inputs": null,
  "outputs": null,
  "icon": null,
  "env": {
    "CPU":["conda create -n anet-cpu python=3.6"],
    "GPU": ["conda create -n anet-gpu2 python=3.6"],
    "Windows-CPU": ["conda create -n anet-win-cpu python=3.6"],
    "Windows-GPU": ["conda create -n anet-win-gpu python=3.6"]
  },
  "requirements": {"CPU":["repo: https://github.com/oeway/Anet-Lite", "tensorflow==1.5", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "GPU": ["repo: https://github.com/oeway/Anet-Lite", "tensorflow-gpu==1.5", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "Windows-CPU":["repo: https://github.com/oeway/Anet-Lite", "tensorflow==1.2", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"],
                    "Windows-GPU": ["repo: https://github.com/oeway/Anet-Lite", "tensorflow-gpu==1.2", "keras==2.2.1", "Pillow", "git+https://www.github.com/keras-team/keras-contrib.git", "pytest"]
   },
   "flags": [],
  "dependencies": ["oeway/ImJoy-Plugins:Im2Im-Dashboard"]
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
        self.output_channels = ['output_'+ch[0] for ch in opt.target_channels]

    def on_batch_end(self, batch, logs):
        self.logs = logs
        api.showStatus('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' ' + str(logs))
        sys.stdout.flush()
        self.dash.updateCallback('onStep', self.step, {'mse': np.asscalar(logs['mean_squared_error']), 'dssim_l1': np.asscalar(logs['DSSIM_L1'])})
        self.step += 1
        if abort.is_set():
            raise Exception('Abort.')

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        self.logs = logs
        api.showProgress(self.epoch/self.total_epoch*100)
        api.showStatus('training epoch:'+str(self.epoch)+'/'+str(self.total_epoch) + ' '+ str(logs))
        xbatch, ybatch = next(self.gen)
        ypbatch = self.model.predict(xbatch, batch_size=1)
        tensor_list = [xbatch, ypbatch, ybatch]
        label = 'Step '+ str(self.step)
        titles = [self.input_channels, self.output_channels, self.target_channels]
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
        api.showStatus("A-net lite successfully initialized.")

    async def setup(self):
        #api.register(name="set working directory", run=self.set_work_dir, ui="set working directory for loading data and saving trained models")
        api.register(name="get example dataset", run=self.download_data, ui="download example data set to your workspace")
        api.register(name="load trained weights", run=self.load_model_weights, ui="load a trained weights for the model")
        api.register(name="train", run=self.train, ui="name:{id:'name', type:'string', placeholder:''}<br>"\
                "epochs:{id:'epochs', type:'number', min:1, placeholder:100}<br>"\
                "steps per epoch:{id:'steps', type:'number', min:1, placeholder:30}<br>"\
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
            targetObj = await api.showFileDialog(root='', type='file')
            weight_path = targetObj.path
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
        targetObj = await api.showFileDialog(title="Please select a working directory (with data)")
        work_dir = targetObj.path
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
        totalsize = len(source)
        self.dash = await api.createWindow(type="Im2Im-Dashboard", name="Anet-lite Prediction", w=25, h=10, data={"display_mode": "all"})

        input_channels = [ch[0] for ch in self._opt.input_channels]
        output_channels = ['output_'+ch[0] for ch in self._opt.target_channels]

        for i in range(int(totalsize/batch_size+0.5)):
            xbatch, paths = next(gen)
            ypbatch = self.model.predict(xbatch, batch_size=batch_size)
            tensor_list = [xbatch, ypbatch]
            label = 'Sample '+ str(i)
            titles = [input_channels, output_channels]
            plot_tensors(self.dash, tensor_list, label, titles)
            count +=batch_size
            for b in range(len(ypbatch)):
                image = ypbatch[b]
                path = paths[b]
                _, name = os.path.split(path)
                output_path = os.path.join(output_dir, name)
                for i in range(image.shape[2]):
                    # im = Image.fromarray(image[:, :, i].astype('float32'))
                    # im.save(output_path+'_'+output_channels[i]+'_output.tif')
                    misc.imsave(output_path+'_'+output_channels[i]+'_output.tif', image[:, :, i].astype('float32'))
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
        abort.clear()
        tensorboard = TensorBoard(log_dir=os.path.join(opt.checkpoints_dir, my.config.name + 'logs'), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True)
        checkpointer = ModelCheckpoint(filepath=os.path.join(opt.checkpoints_dir,  my.config.name + '__model__.hdf5'), verbose=1, save_best_only=True)
        self.model.fit_generator(make_generator(sources['train'], batch_size=opt.batch_size),
                            validation_data=make_generator(sources['valid'], batch_size=opt.batch_size),
                            validation_steps=4, steps_per_epoch=my.config.steps, epochs=epochs, verbose=2, callbacks=[updateUI, checkpointer, tensorboard])
        self.model.save(os.path.join(opt.checkpoints_dir,  my.config.name + '__model__.hdf5'))

        model_config = {}
        model_config['input_size'] = opt.input_size
        model_config['input_channels'] = len(opt.input_channels)
        model_config['target_channels'] = len(opt.target_channels)

        with open(os.path.join(opt.work_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)


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
# if __name__ == '__main__':
#     test_run = ImJoyPlugin()
#     test_run.train_run("")

</script>

<attachment name="model_config_template">
{
  "outputs": [{
    "name": "output",
    "key": "output",
    "type": "image",
    "channels": ["SR"],
    "size": 512,
    "shape": [1, 512, 512, 1]
  }],
  "label": "Unet Model",
  "model_name": "Unet_512x512",
  "url": "",
  "inputs": [{
    "name": "input",
    "key": "input",
    "type": "image",
    "channels": ["SR", "LR"],
    "size": 512,
    "shape": [1, 512, 512, 2],
    "default": 0.0,
    "required": true,
    "hide": false
  }]
}

</attachment>
