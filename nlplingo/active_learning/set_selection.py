import json
import sys
from collections import defaultdict

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    predictions_files = [x.strip() for x in f.readlines()]

selected_predictions = defaultdict(dict)
complete_predictions = defaultdict(dict)

budget = 20000

for predictions_file in predictions_files:
    predictions = json.load(open(predictions_file, 'r'))
    for event_type, examples in predictions.iteritems():
        for example_name, value in examples.iteritems():
            complete_predictions[event_type][example_name] = value

#for (key, value) in sorted(complete_predictions.iteritems(), key=lambda x: len(x[1].keys())):
#    print('{} {}'.format(key, len(value.keys())))

collected = 0
event_type_index = 0
event_types = complete_predictions.keys()
while collected < budget:
    if len(complete_predictions[event_types[event_type_index]].keys()) > 0:
        candidate = sorted(
            complete_predictions[event_types[event_type_index]].iteritems(),
            key=lambda x: x[1]['active_learning_raw_score'],
            reverse=True
        )[0]
        selected_predictions[event_types[event_type_index]][candidate[0]] = candidate[1]
        del complete_predictions[event_types[event_type_index]][candidate[0]]
        collected += 1
    event_type_index += 1
    if event_type_index > len(event_types) - 1:
        event_type_index = 0

for (key, value) in sorted(selected_predictions.iteritems(), key=lambda x: len(x[1].keys())):
    print('{} {}'.format(key, len(value.keys())))

with open(output_file, 'w') as f:
    json.dump(selected_predictions, f, sort_keys=True, indent=4)
