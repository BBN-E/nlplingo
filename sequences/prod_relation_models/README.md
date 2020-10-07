Invocation:

`perl sequences/train.pl train.par -sge`

`input_train_file_bert`: training set json file for Bert model

`input_val_file_bert`: dev set json file for Bert model

`num_epoch_bert`: number of epochs to train Bert mention pooling model 

`ontology_bert`: ontology file for Bert mention pooling model (compatible with `input_train_file_bert` and `input_val_file_bert`


`input_train_file_cnn`: training set json file for cnn model

`input_val_file_cnn`: dev set json file for cnn model

`num_epoch_cnn`: number of epochs to train cnn model 

`ontology_cnn`: ontology file for cnn model (compatible with `input_train_file_cnn` and `input_val_file_cnn`

This sequence trains the Bert mention pooling model and LearnIt-Gigaword model. I highly recommend using at least 2 GPU's for each model, where the GPU is >= p100. Using 2 P100's, the expected training time for LearnIt-Gigaword is around 6 hours, and the Bert mention pooling model takes around 1.5 hours.

Example experiment directory: `/nfs/raid88/u10/users/jcai/expts/delivery/train_ldc_bert_giga_cnn_8_20_20_1`. The trained LearnIt-GigaWord CNN models are in `/nfs/raid88/u10/users/jcai/expts/delivery/train_ldc_bert_giga_cnn_8_20_20_1/train_giga_cnn/train` and the trained BERT mention pooling models are in `/nfs/raid88/u10/users/jcai/expts/delivery/train_ldc_bert_giga_cnn_8_20_20_1/train_ldc_bert/train`.
