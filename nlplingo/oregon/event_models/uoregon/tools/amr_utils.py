from nlplingo.oregon.event_models.uoregon.tools.utils import *
import xml.etree.ElementTree as ET


class Predicate:
    def __init__(self, predicate_node):
        self.node = predicate_node
        self.lemma = predicate_node.attrib['lemma'] if 'lemma' in predicate_node.attrib else ''
        self.roleset_list = [RoleSet(roleset_node) for roleset_node in predicate_node.findall('roleset')]


class RoleSet:
    def __init__(self, roleset_node):
        self.node = roleset_node
        self.id = roleset_node.attrib['id']
        self.name = roleset_node.attrib['name']
        # self.aliases = [Alias(alias_node) for alias_node in roleset_node.findall('aliases')[0].findall('alias')]
        # self.roles = [Role(role_node) for role_node in roleset_node.findall('roles')[0].findall('role')]
        self.example_list = [Example(example_node) for example_node in roleset_node.findall('example') if
                             Example(example_node).is_valid]


class Example:
    def __init__(self, example_node):
        self.node = example_node
        self.name = example_node.attrib['name']
        self.text = example_node.findall('text')[0].text
        self.anchors = [rel.text for rel in example_node.findall('rel') if rel.text and len(rel.text.strip())]
        self.agents = [arg.text for arg in example_node.findall('arg') if
                       arg.attrib['n'] == '0' and arg.text and len(arg.text.strip())]
        self.patients = [arg.text for arg in example_node.findall('arg') if
                         arg.attrib['n'] == '1' and arg.text and len(arg.text.strip())]
        # ************ test if anchors, agents and patients actually appear in the sentence ************
        self.is_valid = True
        # print('===============================')
        # print(self.text)
        # print(self.anchors)
        # print(self.agents)
        # print(self.patients)
        for anchor in self.anchors:
            if anchor not in self.text:
                self.is_valid = False
                break
        for agent in self.agents:
            if agent not in self.text:
                self.is_valid = False
                break

        for patient in self.patients:
            if patient not in self.text:
                self.is_valid = False
                break


# class Alias:
#     def __init__(self, alias_node):
#         self.node = alias_node
#         self.pos = alias_node.attrib['pos']
#         self.text = alias_node.text
#
#
# class Role:
#     def __init__(self, role_node):
#         self.node = role_node
#         self.descr = role_node.attrib['descr']
#         self.f = role_node.attrib['f']
#         self.n = role_node.attrib['n']
#         self.vnrole_list = [VnRole(vnrole_node) for vnrole_node in role_node.findall('vnrole')]
#
#
# class VnRole:
#     def __init__(self, vnrole_node):
#         self.node = vnrole_node
#         self.vncls = vnrole_node.attrib['vncls']
#         self.vntheta = vnrole_node.attrib['vntheta']


def get_examples_from_frame(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    predicate = Predicate(root[0])
    examples = []
    for roleset in predicate.roleset_list:
        for ex in roleset.example_list:
            examples.append(
                (ex.text, ex.anchors, ex.agents, ex.patients)
            )
    # for example in examples:
    #     print(example)
    return examples


def get_examples_from_propbank(corpus_path):
    frame_files = [file for file in get_files_in_dir(dir_path=corpus_path, extension='xml')]
    examples = []
    for progress, frame_file in enumerate(frame_files):
        print('{}/{}'.format(progress, len(frame_files)))
        examples += get_examples_from_frame(xml_path=frame_file)
    print('Propbank #examples: {}'.format(len(examples)))
    write_json(examples, write_path=os.path.join(WORKING_DIR, 'datasets', 'AMR', 'arg_examples.propbank.json'))


if __name__ == '__main__':
    # get_examples_from_frame(xml_path=os.path.join(WORKING_DIR,
    #                                               'datasets/AMR/abstract_meaning_representation_amr_2.0/data/frames/propbank-frames-xml-2016-03-08/abandon.xml'))
    get_examples_from_propbank(
        corpus_path=os.path.join(WORKING_DIR,
                                 'datasets/AMR/abstract_meaning_representation_amr_2.0/data/frames/propbank-frames-xml-2016-03-08')
    )
