pythonpath=/d4m/nlp/releases/text-open/R2020_04_15/src/python/:/d4m/nlp/releases/nlplingo/R2020_05_21

CUDA_VISIBLE_DEVICES=0 env KERAS_BACKEND=tensorflow /d4m/better/software/python/singularity/bin/singularity-python.sh -i linux_cuda8_cudnn5 -v /nfs/raid88/u10/users/ychan-ad/miniconda3/envs/t -l "$pythonpath" /d4m/nlp/releases/nlplingo/R2020_05_21/nlplingo/tasks/train_test.py --params train.params --mode train_trigger_from_file

