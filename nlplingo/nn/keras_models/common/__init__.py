from __future__ import absolute_import

from nlplingo.nn.keras_models.common.layers import WeightedHierarchyLayer
from nlplingo.nn.keras_models.common.losses import contrastive_loss
from nlplingo.nn.keras_models.common.losses import masked_bce


keras_custom_objects = {
    u'WeightedHierarchyLayer': WeightedHierarchyLayer,
    u'masked_bce': masked_bce,
    u'contrastive_loss': contrastive_loss
}
