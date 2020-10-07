from abc import ABC, abstractmethod

class Datapoint(ABC):

    def __init__(self):
        """
        A Datapoint is the unit of serialization.  It is the abstract base class
        for candidates of all tasks.
        """
        pass

    def to_data_dict(self, features):
        """
        :param features:  one of the nlplingo.tasks.*.feature.*Feature objects
        :rtype: dict[str:numpy.ndarray]
        """

        d = dict()
        if features is not None:
            for feature in features.activated_features:
                d[feature] = getattr(self, feature)
        d['label'] = self.label
        return d

    """
    @abstractmethod
    def _allocate_arrays(self, hyper_params, vector_size, none_token_index, features):
        pass
    """