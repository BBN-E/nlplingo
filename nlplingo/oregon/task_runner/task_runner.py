import json
import logging
import subprocess
import sys

from enum import Enum

COMMANDS_JSON = "/code/docker/commands.json"
TASK_JSON = "/app/tasks.json"
LOG_FILE = "/app/tasks.log"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# These are ordered (i.e., train-from-data has to happen before abstract-events)
TASKS = ['train-from-data', 'abstract-events']


# This is a convenience function that maps from the tasks defined in
# /app/tasks.json to actual commands with arguments.  The commands and arguments
# are stored in the path pointed to by the COMMANDS variable.
def load_commands():
    """
    Load the list of command line processes to start for a given task.
    """
    task_commands = {}
    with open(COMMANDS_JSON, 'r') as fin:
        data = json.load(fin)
        for task in data:
            if task not in TASKS:
                raise RuntimeError(
                    f'Unrecognized task in commands.json: {task}')
            task_commands[task] = [data[task].split(';')[k].split() for k in range(len(data[task].split(';')))]
    logging.info(f'Loaded {len(task_commands.keys())} task commands.')
    return task_commands


# Per the evaluation plan, /app/tasks.json will have a format like:
# {"execution-start-timestamp": "2018-07-11 18:58:23",
#  "task-set":
#    ["train-from-data",
#     "abstract-events",
#     "basic-events",
#     "granular-events",
#     "document-relevance",
#     "human-in-the-loop"]}
# For now, we only expect to see "train-from-data" and "abstract-events".
# Note that "train-from-data" really means "make use of an incremental training
# data set by adding to your existing training data".
def parse_tasks_json(json_file):
    """
    Extract tasks from json task file.
    """
    task_commands = load_commands()
    with open(json_file, 'r') as fin:
        data = json.load(fin)
        # Make sure we recognize all the tasks
        for task in data['task-set']:
            if task not in task_commands:
                raise NotImplementedError(
                    f'Task {task} is not implemented.')
        # Run tasks in order
        for task in TASKS:
            if task in data['task-set']:
                for command in task_commands[task]:
                    run_task(task, command)


def run_task(task, task_command):
    logging.info(f'Starting task: {task} with command '
                 f'{str(task_command)}')
    ret = subprocess.call(task_command)
    if ret == 0:
        logging.info("Completed command: " + task)


def main():
    """
    Execute the tasks specified in tasks.json.

    tasks - json file
    """
    parse_tasks_json(TASK_JSON)


if __name__ == "__main__":
    main()
