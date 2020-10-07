from collections import defaultdict
from abc import ABC, abstractmethod

class ExampleGenerator(ABC):

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        self.event_domain = event_domain
        self.params = params
        self.extractor_params = extractor_params
        self.hyper_params = hyper_params
        self.statistics = defaultdict(int)
        self.train_dev_test_mode = None
        self.decode_mode = self.hyper_params.decode_mode

    def examples_to_data_dict(self, examples, features):
        """
        Organize examples into a dictionary which indexes by feature name.
        :type examples: list[nlplingo.tasks.common.Datapoint
        :type features: nlplingo.tasks.feature.feature_setting.FeatureSetting
        """
        data_dict = defaultdict(list)
        for example in examples:
            example_data = example.to_data_dict(features)
            for k, v in example_data.items():
                data_dict[k].append(v)
        return data_dict

    @abstractmethod
    def generate(self, docs):
        """
        Generate candidates from nlplingo docs.
        :param docs: list[nlplingo.text.text_theory.Document]
        :return:
        """
        pass