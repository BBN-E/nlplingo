
#### Coding conventions

These are minimal, as follows:
* We use PyCharm as our UI for coding. Set your tab stops to 4 spaces.
* Class names should be mixed case, e.g. ```EventExtractionModel```
* Variable names, function names, should be lower case, e.g. ```calculate_accuracy```
* Although Python is free type, but adding type hints in your code allows code completion, and is useful to whoever is reading your code, e.g.

```
def add_entity_mention(self, entity_mention):
    """:type entity_mention: nlplingo.text.text_span.EntityMention"""
```

You can also add type hints after variable declaration, e.g.:
```
self.entity_mentions = []
""":type: list[nlplingo.text.text_span.EntityMention]"""
```

#### Setting up nlplingo

nlpLingo now uses Tensorflow as backend.

Setting up tensorflow GPU anaconda:
```
conda create --name p3-gpu
source activate p3-gpu
conda install -c anaconda python=3.6.8
conda install -c anaconda tensorflow-gpu=1.12 (use 'tensorflow=1.12' for CPU)
conda install keras=2.2.4
conda install -c anaconda future
conda install spacy
conda install -c anaconda simplejson (else you will get an error "simplejson has no attribute loads" when you try to do the 'spacy download')
python -m spacy download en
conda install -c anaconda pytorch (use '-c anaconda pytorch-cpu' for CPU)
conda install -c conda-forge python-crfsuite (optional, just for NER)
```

Note: tensorflow seems to need GLIBC 2.12, while torch needs GLIBC 2.14. To add GLIBC 2.14: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/glibc-2.14/lib`
But if you do that, then tensorflow will not work, while pytorch will work

Testing that your GPU install is working:
```
import tensorflow as tf
tf.test.is_gpu_available()

import torch
torch.cuda.is_available()
```

Do ```export KERAS_BACKEND=tensorflow```

Or set here permanently ```/nfs/raid84/u12/ychan/anaconda2/envs/p3-gpu/etc/conda/activate.d/keras_activate.sh```

If you want to add the python env to your jupyter kernel:
```
conda install -c anaconda ipykernel
conda install -c anaconda jupyter
ipython kernel install --name p3-gpu --user
jupyter kernelspec list (shows the python kernels that you can use for jupyter)
```

The following that uses Theano as the backend is now deprecated.
```
conda create --name nlplingo python=2.7.12 numpy=1.12.1 scipy=0.19.0 openblas=0.2.19 theano=0.9.0 keras=2.0.2 spacy=1.8.2
source activate nlplingo
python -m spacy download en
conda install -c anaconda future
conda install -c anaconda scikit-learn
conda install -c conda-forge jsonpickle
conda install -c conda-forge python-crfsuite
conda install -c anaconda mkl-service

export env KERAS_BACKEND=theano
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda-8.0/lib64
export CUDA_ROOT=/opt/cuda-8.0
```

In ~/.keras/keras.json , set "backend" : "theano" 


#### Running nlplingo

You need to point PYTHONPATH to 2 locations:
- dir that contains serifxml.py
- dir where nlplingo resides

This is an example command:
PYTHONPATH=/home/ychan/anaconda2/envs/bbn_env/lib/python2.7/site-packages/serif:/home/ychan/repos/nlplingo python /home/ychan/repos/nlplingo/nlplingo/event/train_test.py --params x.params --mode test_argument

Here are the available modes for train_test.py: 
- train_trigger
- test_trigger
- train_argument
- test_argument
- decode_trigger_argument

Comments:
- The scores are written to *.score files
- The dataset is divided into train, dev, test. train_*.score files report evaluation on dev set. test_*.score files report evaluation on test set.
- We'll normally do hyper-parameters search over (cnn.neighbor_dist, *.positive_weight, *.batch_size, *.num_feature_maps, *.epoch)
- Note that each run will produce different scores/results (e.g. due to drop-out). So we also normally do a few runs per hyper-params setting, and take the one which performs best on dev.
- The hdf and pickle are model files.

