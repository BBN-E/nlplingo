import os
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *


def get_rsync_files(path):
    ignore_list = [
        'python/clever/event_models/uoregon/tools/stanford_resources',
        'python/clever/event_models/uoregon/tools/bert_resources',
        'python/clever/event_models/uoregon/datasets',
        'python/clever/event_models/uoregon/models/pipeline/_01/output/predictions',
        'python/clever/event_models/uoregon/logs'
    ]

    for ignore_dir in ignore_list:
        if ignore_dir in path:
            return []

    if os.path.isfile(path):
        return [path]
    else:
        objects = [os.path.join(path, object_name) for object_name in os.listdir(path)]
        subpaths = []
        for obj in objects:
            subpaths += get_rsync_files(obj)
        return subpaths


def create_uo_rsync_list():
    start_dir = './python/clever/'
    all_uo_files = get_rsync_files(start_dir)
    all_uo_files = ['.' + uo_file for uo_file in all_uo_files if '.pyc' != uo_file[-len('.pyc'):]]

    all_uo_files += ['../docker/commands.json']
    with open('./docker/rsync.list', 'w') as f:
        f.write('\n'.join(all_uo_files))


if __name__ == '__main__':
    create_uo_rsync_list()
