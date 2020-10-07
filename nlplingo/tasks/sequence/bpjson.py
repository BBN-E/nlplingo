
def document_prediction_to_bp_json(documents, corpus_id):
    """
    :type document: list[DocumentPrediction]
    """
    bp = dict()
    bp['corpus-id'] = corpus_id
    bp['format-type'] = 'bp-corpus'
    bp['format-version'] = 'v8f'
    bp['entries'] = dict()

    for document in documents:
        for sentence in document.sentences.values():
            entry_id = sentence.id
            bp['entries'][entry_id] = dict()
            bp['entries'][entry_id]['doc-id'] = entry_id.split('_')[0]
            bp['entries'][entry_id]['sent-id'] = entry_id.split('_')[1]
            bp['entries'][entry_id]['entry-id'] = entry_id
            bp['entries'][entry_id]['segment-type'] = 'sentence'
            bp['entries'][entry_id]['segment-text'] = sentence.text

            bp['entries'][entry_id]['annotation-sets'] = dict()
            bp['entries'][entry_id]['annotation-sets']['abstract-events'] = dict()
            bp['entries'][entry_id]['annotation-sets']['abstract-events']['events'] = dict()
            bp['entries'][entry_id]['annotation-sets']['abstract-events']['span-sets'] = dict()

            spans = set()
            # first collect all the spans from triggers and arguments
            for event in sentence.events.values():
                spans.add(event.trigger.text)
                spans.update(argument.text for argument in event.arguments.values())

            span_to_id = dict()
            for i, span in enumerate(sorted(spans)):
                span_to_id[span] = 'ss-{}'.format(str(i+1))

            for span in span_to_id:
                span_id = span_to_id[span]
                bp['entries'][entry_id]['annotation-sets']['abstract-events']['span-sets'][span_id] = dict()
                bp['entries'][entry_id]['annotation-sets']['abstract-events']['span-sets'][span_id]['ssid'] = span_id
                bp['entries'][entry_id]['annotation-sets']['abstract-events']['span-sets'][span_id]['spans'] = []
                span_d = dict()
                span_d['hstring'] = span
                span_d['string'] = span
                bp['entries'][entry_id]['annotation-sets']['abstract-events']['span-sets'][span_id]['spans'].append(span_d)

            for i, event in enumerate(sentence.events.values()):
                event_d = dict()
                event_id = 'event{}'.format(str(i+1))

                assert event.trigger.text in span_to_id
                trigger_id = span_to_id[event.trigger.text]
                event_d['anchors'] = trigger_id
                event_d['eventid'] = event_id

                assert len(event.trigger.labels) == 1
                event_types = list(event.trigger.labels.keys())[0].split('.')
                assert len(event_types) == 2
                event_d['helpful-harmful'] = event_types[0]
                event_d['material-verbal'] = event_types[1]

                event_d['agents'] = []
                event_d['patients'] = []

                for argument in event.arguments.values():
                    assert argument.text in span_to_id
                    argument_id = span_to_id[argument.text]
                    assert len(argument.labels) == 1
                    argument_role = list(argument.labels.keys())[0].lower()
                    if argument_role == 'agent':
                        event_d['agents'].append(argument_id)
                    elif argument_role == 'patient':
                        event_d['patients'].append(argument_id)

                bp['entries'][entry_id]['annotation-sets']['abstract-events']['events'][event_id] = event_d

    return bp
