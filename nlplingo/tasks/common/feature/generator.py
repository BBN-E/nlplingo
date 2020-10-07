from abc import ABC, abstractmethod

class FeatureGenerator(ABC):
    def __init__(self, extractor_params, hyper_params, feature_setting):
        """

        :param extractor_params: dict
        :param hyper_params: nlplingo.nn.hyperparameters.HyperParameters
        :param feature_setting: nlplingo.tasks.common.feature.feature_setting.FeatureSetting
        """
        self.extractor_params = extractor_params
        self.hyper_params = hyper_params
        self.feature_setting = feature_setting

    def assign_example(self, example, feature_name, args):
        """
        A helper function to aid in adding features to the example.
        Explicitly assign an attribute of the datapoint, with its corresponding feature generation function in the datapoint.
        By convention, the datapoint's attribute is the same as the feature generation function name within the datapoint.

        The second argument is the feature name, which defines the feature generation function.
        The third argument is any arguments (in the right order) passed to the feature generation function.
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :param feature_name: str
        :param args: list of variable size that is passed into the feature generation function as arguments
        :return:
        """
        assert(hasattr(example, feature_name) and callable(getattr(example, feature_name))) # assert that the feature is implemented
        if feature_name in self.feature_setting.activated_features:
            setattr(example, feature_name, getattr(example, feature_name)(*args))

    def populate(self, examples):
        """
        Populate a set of examples with features.
        :param examples: list[nlplingo.tasks.common.datapoint.Datapoint]
        :return:
        """
        for example in examples:
            self.populate_example(example)

    @abstractmethod
    def populate_example(self, example):
        """
        Populate the candidate example with features.
        :param example: nlplingo.tasks.common.datapoint.Datapoint
        :return:
        """
        pass