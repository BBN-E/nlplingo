from collections import OrderedDict

class FeatureSetting(object):
    def __init__(self, activated_features):
        """
        An object used to store activated features (as strings).
        :param activated_features: list[str]
        """
        ordered_indices = list(range(0,len(activated_features)))
        self.activated_features = OrderedDict(zip(activated_features, ordered_indices))

        """
        Might be useful later on
        
        self.feature_order = {}
        self.tensor_types = []
        for feature_idx, feature in enumerate(self.feature_strings):
            self.feature_order[feature] = feature_idx
            if 'sent_re_type' in self.usable_features[feature]:
                self.tensor_types.append(self.usable_features[feature]['sent_re_type'])
            else:
                self.tensor_types.append('long')
        """