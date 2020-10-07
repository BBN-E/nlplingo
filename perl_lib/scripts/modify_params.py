import argparse
import json
import logging
import os


logger = logging.getLogger(__name__)


def __parse_setup():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--base_params_file',
        type=str, 
    )
    parser.add_argument(
        '--trigger_params_file', 
        type=str
    )
    parser.add_argument(
        '--argument_params_file',
        type=str
    )
    parser.add_argument(
        '--output_params_file',
        type=str,
        required=True
    )
    return parser


def __main(args):

    if args.base_params_file is None:
        logger.info("No base params file; using supplied params file instead "
                    "(trigger if available)")
        if args.trigger_params_file:
            base_params_data = json.load(open(args.trigger_params_file, 'r'))
        elif args.argument_params_file:
            base_params_data = json.load(open(args.argument_params_file, 'r'))
        else:
            raise IOError("Must supply either base params or new params!")
    else:
        base_params_data = json.load(open(args.base_params_file, 'r'))
    base_params_data['extractors'] = []
    
    if args.trigger_params_file:
        trigger_data = json.load(open(args.trigger_params_file, 'r'))
        base_params_data['extractors'].append(trigger_data['extractors'][0])
    if args.argument_params_file:
        argument_data = json.load(open(args.argument_params_file, 'r'))
        base_params_data['extractors'].append(argument_data['extractors'][0])
    json.dump(
        base_params_data,
        open(args.output_params_file,'w'),
        sort_keys=True,
        indent=4,
    )


if __name__ == '__main__':
    __main(__parse_setup().parse_args())
