import json
from nlplingo.common.utils import IntPair
from nlplingo.text.text_theory import Document as lingoDoc
from nlplingo.text.text_span import Sentence
from nlplingo.text.text_span import Token
from nlplingo.text.text_theory import Event
from nlplingo.text.text_theory import Entity
from nlplingo.text.text_span import EntityMention

from nlplingo.text.text_span import Anchor
from nlplingo.text.text_span import EventArgument


class Instance(object):
    @staticmethod
    def convert_pos(pos):
        return pos  # pass through for now

    @staticmethod
    def __recover_offsets(tokens, start_index, end_index):
        start = sum([len(x) for x in tokens[:start_index]]) + \
            len(tokens[:start_index])
        end = sum([len(x) for x in tokens[:end_index + 1]]) + \
            len(tokens[:end_index + 1])
        return start, end

    @classmethod
    def to_tokens(cls, instance_dict):
        ret = []
        end_ptr = 0
        for i, (t, pos) in enumerate(
                zip(
                    instance_dict['token'],
                    instance_dict['stanford_pos']
                )
        ):
            if i != 0:
                end_ptr += 1
            t_text = t
            t_start = end_ptr
            t_end = end_ptr + len(t)
            end_ptr = t_end
            t_pos_tag = cls.convert_pos(pos)
            # we do a +1 because this has been the assumption in nlplingo
            ret.append(
                Token(
                    IntPair(
                        t_start,
                        t_end + 1
                    ),
                    i,
                    t_text,
                    lemma=None,
                    pos_tag=t_pos_tag
                )
            )
        return ret

    # @classmethod
    # def to_lingo_docs(cls, filepath):
    #     instances = json.load(open(filepath, 'r'))
    #     docs = []
    #     for instance_dict in instances:
    #         docs.append(
    #             cls.to_lingo_doc(instance_dict)
    #         )
    #     return docs

    @classmethod
    def process_entities(cls, doc, instance_dict):
        id_counter = 0
        obj_index = instance_dict['obj_start']
        head_index = min(int(instance_dict['stanford_head'][obj_index]), len(instance_dict['token'])-1)
        ner = instance_dict['obj_type']
        if ner == 'TME':
            ner = 'Time'
        elif ner == 'VAL':
            ner = 'Value'
        entity_id = instance_dict['id'] + "-0"
        id_counter += 1
        entity = Entity(entity_id)
        mention_id = entity_id + '-0'
        mention_type = 'UNK'
        text = instance_dict['token'][head_index]  # head
        start, end = cls.__recover_offsets(
            instance_dict['token'],
            head_index,
            head_index
            )
        em = EntityMention(
            mention_id,
            IntPair(start, end),
            text,
            ner,
            entity,
            mention_type
        )
        doc.add_entity_mention(em)
        entity.mentions.append(em)
        doc.add_entity(entity)

    @classmethod
    def process_events(cls, doc, instance_dict):
        event_id = instance_dict['id']
        event_type = instance_dict['subj_type']
        mention_id = event_id + '-0'
        event = Event(mention_id, event_type)
        anchor_text = ' '.join(
            instance_dict['token'][int(instance_dict['subj_start']):int(instance_dict['subj_end'])+1]
        )
        start, end = cls.__recover_offsets(
            instance_dict['token'],
            int(instance_dict['subj_start']),
            int(instance_dict['subj_end'])
        )
        event.add_anchor(
            Anchor(
                mention_id+'-trigger',
                IntPair(start, end),
                anchor_text,
                event_type
            )
        )
        arg_id = mention_id + '-0'
        arg_role = instance_dict['relation']
        if arg_role.startswith('Time-'):
            arg_role = 'Time'

        arg_em = doc.get_entity_mention_with_id(arg_id)
        event_arg = EventArgument(
            '{}-a{}'.format(mention_id, event.number_of_arguments()),
            arg_em,
            arg_role
        )
        event.add_argument(event_arg)
        doc.add_event(event)

    @classmethod
    def to_lingo_doc(cls, filepath):
        instance_dict = json.load(open(filepath, 'r'))
        #print(instance_dict)
        docid = instance_dict['id']
        lingo_doc = lingoDoc(docid)

        tokens = cls.to_tokens(instance_dict)
        st_text = ' '.join(instance_dict['token'])
        st_start = 0
        st_end = len(st_text)
        st_index = 0
        s = Sentence(docid, IntPair(st_start, st_end+1), st_text, tokens, st_index)
        lingo_doc.add_sentence(s)
        cls.process_entities(lingo_doc, instance_dict)
        cls.process_events(lingo_doc, instance_dict)
        return lingo_doc
