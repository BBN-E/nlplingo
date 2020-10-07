from __future__ import absolute_import

from keras.engine.topology import Layer
from keras.layers import Input

# Assuming theano backend
from keras import backend as K

import numpy as np


######################################################
#  Custom Keras layers currently in use by NLPLingo  #
######################################################


class WeightedHierarchyLayer(Layer):

    def __init__(self, out_dim, prior, previous_dim, **kwargs):
        self.output_dim = out_dim
        self.prior_ndarray = prior
        self.previous_dim = previous_dim
        self.S = None
        self.V = None
        self.trainable_weights = []
        super(WeightedHierarchyLayer, self).__init__(**kwargs)

    def _weight_variable(self):
        shape = (self.output_dim, self.previous_dim)
        initial_array = np.random.uniform(-0.01, 0.01, size=shape)
        tensor = K.variable(value=initial_array, dtype=u'float32')
        return Input(tensor=tensor, dtype=u'float32')

    def build(self, in_shape):
        self.S = K.constant(self.prior_ndarray, dtype=u'float32')
        self.V = self._weight_variable()
        self.trainable_weights = [self.V]
        super(WeightedHierarchyLayer, self).build(in_shape)

    def call(self, x):
        W = K.transpose(K.dot(self.S, self.V))
        return K.dot(x, W)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim
