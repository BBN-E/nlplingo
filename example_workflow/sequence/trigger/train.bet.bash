python=/nfs/raid88/u10/users/ychan-ad/miniconda3/envs/t/bin/python
pythonpath=/d4m/nlp/releases/nlplingo/R2020_05_21:/nfs/raid88/u10/users/ychan/repos/text-open/src/python
script=/d4m/nlp/releases/nlplingo/R2020_05_21/nlplingo/tasks/train_test.py

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$pythonpath" "$python" "$script" --params train.bet.params --mode train_trigger_from_file
