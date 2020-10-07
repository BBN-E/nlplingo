Invocation:

`perl runjob_scripts/canonical.pl params/canonical.par -sge`

Configurable parameters in params/canonical.par :

`stages_to_run`: There are two: `preprocess` is for featurizing and serializing nlplingo Datapoints to disk; `train` is for actually training a model.

`num_of_batches_global` : controls number of batches for serializing nlplingo Datapoints onto disk

`tasks` : a comma-separated list of tasks to perform train/test over

`train_canonical_template_dir` : location of nlplingo training config files for each task. By convention, each config file is named `task.json` where `task` is the task name

`test_canonical_template_dir` : location of nlplingo testing config files for each task. By convention, each config file is named `task.json` where `task` is the task name

`decode_canonical_template_dir` : location of nlplingo decoding config files for each task. By convention, each config file is named `task.json` where `task` is the task name

`in_memory`: Set to 1 if you want to do everything in memory (e.g. skip the serialization to disk step; featurize everything in memory), 0 otherwise, for training the model.

TODO: For decoding, adapt parallelization procedure from Hume
