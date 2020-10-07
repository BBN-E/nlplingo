import random
import six
import abc




@six.add_metaclass(abc.ABCMeta)
class ActiveBatchSelector(object):
    @abc.abstractmethod
    def select(
            self,
            trigger_model,
            subset,
            train_data_label,
            train_data_list,
            budget
    ):
        pass


class RandomActiveBatchSelector(ActiveBatchSelector):

    class Factory(object):
        
        @staticmethod
        def create(self, activebatchselector_params):
            return RandomActiveBatchSelector(activebatchselector_params)

    def __init__(self, activebatchselector_params):
        self.__activebatchselector_params = activebatchselector_params

    def select(
            self,
            trigger_model,
            subset,
            train_data_label,
            train_data_list,
            budget
    ):
        """
        This function randomly selects new examples to add for training.
        The number of examples is determined by `budget`.

        :param trigger_model: neural network model object for nlplingo
        :type trigger_model: nlplingo.nn.trigger_model.TriggerModel
        :param subset: List of indexes for the current training set
        :type subset: list
        :param train_data_label: Binary matrix of labels for entire training set [examples x n_labels]
        :type train_data_label: numpy.array
        :param train_data_list: List of feature matrices, one entry for each feature. The feature matrices are arranged
        line this: [examples x 1 x feature_dim]
        :type train_data_list: list[numpy.array]
        :param budget: Number of examples to select
        :type budget: int
        :return: List of new examples to select.
        :rtype: list
        """
        n_total_samples = train_data_label.shape[0]
        candidates = [x for x in range(n_total_samples) if x not in subset]
        return random.sample(candidates, budget) \
            if budget < len(candidates) \
            else candidates


class ActiveBatchSelectorFactory(object):
    factories = {
        'RandomActiveBatchSelector': RandomActiveBatchSelector.Factory(),
    }

    @staticmethod
    def createActiveBatchSelector(id, activebatchselector_params):
        if id in ActiveBatchSelectorFactory.factories:
            return ActiveBatchSelectorFactory.factories[id].create(activebatchselector_params)
        else:
            raise RuntimeError(
                'Active Batch Selector type not supported: {}'.format(
                    id
                )
            )
