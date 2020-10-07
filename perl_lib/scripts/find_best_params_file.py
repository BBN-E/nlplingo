import argparse
import json
import os

def __parse_setup():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--input_score_list', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--output_params_file', 
        type=str, 
        required=True
    )
    return parser

def __main(args):
    score_files = [x.strip() for x in open(args.input_score_list, 'r').readlines()]
    best_F1 = 0.0
    best_score_file = None
    for score_file in score_files:
        top_line = open(score_file,'r').readline().strip()
        F1 = float(top_line.split(",")[-1])
        if F1 > best_F1 or (best_score_file is None):
            best_score_file = score_file
            best_F1 = F1
    if best_score_file is not None:
        dir_path = os.path.dirname(os.path.abspath(best_score_file))

        # walk (in case we used k-folds CV) -- assume all CV params are valid
        for parent, dirs, files in os.walk(dir_path):
            if 'params.json' in files:
                params_file_in = os.path.join(parent, 'params.json')
                break

        json_data = json.load(open(params_file_in,'r'))
        json.dump(
            json_data, 
            open(args.output_params_file,'w'),
            sort_keys=True,
            indent=4,
        )
    else:
        raise RuntimeError('No score file selected!')

if __name__ == '__main__':
    __main(__parse_setup().parse_args())
    


