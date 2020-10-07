#### This uses transformers sequence model to train a trigger extractor

train.bet.bash : 
* The bet-rtx8000-100 machine provides MUCH faster training. This script could only be executed on the bet-rtx8000-100 machine.
* This script uses train.bet.params : Change 'model_file' and 'output_dir' to point to your own directory.

train.bash : 
* If you do not wish to do training on bet-rtx8000-100 machine, then use this script, which provides a singularity. The huggingface transformers XLMR training code relies on having installed PyTorch1.4, which relies on having glibc 2.16 on your machine. bet-rtx8000-100 has glibc 2.17. Other machines typically has glibc 2.12.
* For instance, you can run this script on effect100.
* This script uses train.params. Change 'model_file' and 'output_dir' to point to your own directory.

#### This uses already trained trigger and argument sequence models, to do decoding
decode.bash : 
* This script uses singularity, so you can run this on e.g. effect100 machine.
* This script uses decode.params. Change 'output_bp' and 'predictions_file' to point to your own directory.


The transformers sequence modeling code in NLPLingo was developed within the context of the BETTER project. Hence the above examples take BP-JSON file and corresponding SerifXMLs as input. We plan to provide future examples on other dataset e.g. ACE.

