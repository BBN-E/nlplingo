from __future__ import absolute_import

# Assuming theano backend
from keras import backend as K


#####################################
#  custom loss functions for Keras  #
#####################################


def masked_bce(y_true, y_pred):
    """
    Same as binary cross-entropy loss, but ignores samples where label == -1
    https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/
    """
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))
