from collections import defaultdict
import heapq as hq
import os,re,sys,serifxml3
import xml.etree.ElementTree as etree


log_dir = "/d4m/ears/expts/46693_event_trigger_IR_10.copy/expts/b50e50hl512_512_512lr5e-05mdsc5mps200mtns1000mtes200000mtsc50mtrs1000000mtrsc10/"
input_file = log_dir + "test.collated.log"
query_information_file = log_dir + "information.txt"
output_file = input_file + ".parsed"

top_n = 50
LEFT_CONTEXT_CHAR_SIZE = 70
RIGHT_CONTEXT_CHAR_SIZE = 70
filelists = ["/nfs/raid88/u10/users/bmin/experiments/ace_sgm.filelist"]

doc_id_to_sgm_content = dict()

def extract_text(node, all_text):
    u""" Return list of text from XML element subtree.
    Python 2 version"""
    tag = node.tag
    if not isinstance(tag, str) and tag is not None:
        return
    text = node.text
    if text:
        # print('A:{}[{}]'.format(str(len(text)), str(text)))
        # text = text.replace('\n', ' ')
        all_text.append(str(text))
    for e in node:
        extract_text(e, all_text)
        text = e.tail
        if text:
            # print('B:{}[{}]'.format(str(len(text)), text))
            # text = text.replace('\n', ' ')
            all_text.append(text)
    return all_text


def get_event_mention_info(event_mention_id):
    char_offs = event_mention_id.split("_")[-1]
    doc_id = "_".join(event_mention_id.split("_")[:-1])
    start_char_off = int(char_offs.split("-")[0])
    end_char_off = int(char_offs.split("-")[-1])
    return doc_id,start_char_off,end_char_off


def read_sgm_file_id_and_content(filepath):
    docid = filepath.split("/")[-1].split(".sgm")[0]

    sgml_tree = etree.parse(filepath)
    sgml_root = sgml_tree.getroot()
    all_text = []
    text_list = extract_text(sgml_root, all_text)

    data =  ''.join(text_list)

    return docid, data


def is_span_overlap(ev_start_char, ev_end_char, sent_star_char, sent_end_char):
    if ev_start_char>=sent_star_char and ev_end_char<=sent_end_char:
        return True
    else:
        return False


def santize(text):
    return text.replace("\t", " ").replace("\n", " ")


def get_event_mention_string(event_mention_id):

    doc_id, ev_start_char_off, ev_end_char_off = get_event_mention_info(event_mention_id)

    left_context_start = ev_start_char_off-LEFT_CONTEXT_CHAR_SIZE
    left_context_end = ev_start_char_off

    right_context_start = ev_end_char_off
    right_context_end = ev_end_char_off + RIGHT_CONTEXT_CHAR_SIZE

    left_context_string = doc_id_to_sgm_content[doc_id][left_context_start:left_context_end]
    trigger_string = doc_id_to_sgm_content[doc_id][ev_start_char_off:ev_end_char_off]
    right_context_string = doc_id_to_sgm_content[doc_id][right_context_start:right_context_end]

    return santize(left_context_string) + "[" + santize(trigger_string) + "]" + santize(right_context_string)


def main():
    with open(input_file, "r") as f_in:
        lines = f_in.readlines()

    query2top_sents = defaultdict(list)
    for line in lines:
        li = line.split()
        query = li[0].split("=")[1]
        similarity = float(li[1].split("=")[1])
        trigger_type = li[3].split("=")[1]
        sent_id = li[4].split("=")[1]
        entry = (similarity, trigger_type, sent_id)
        query2top_sents[query].append(entry)

    query2examples = defaultdict(list)
    with open(query_information_file, "r") as f_in:
        lines = f_in.readlines()
    for line in lines:
        li = line.split()
        if li[0] == "QUERY":
            query2examples[li[1]].append(li[2])

    cnt = 0
    for filelist in filelists:
        with open(filelist) as fp:
            for idx,i in enumerate(fp):
                i = i.strip()

                sgm_path = i

                docid, sgm_file_content = read_sgm_file_id_and_content(sgm_path)
                doc_id_to_sgm_content[docid] = sgm_file_content
                print("Loading({}) {}".format(cnt, sgm_path))
                cnt += 1

    with open(output_file, "w") as f_out:
        for query in query2top_sents.keys():
            f_out.write('***QUERY TYPE {} is defined by:\n'.format(query))
            for example in query2examples[query]:
                example_str = get_event_mention_string(example)
                f_out.write('{}\n'.format(example_str))
            hq.heapify(query2top_sents[query])
            top_sents = hq.nlargest(top_n,query2top_sents[query])
            for sent in top_sents:
                sent_str = get_event_mention_string(sent[2])
                f_out.write('query-type={} similarity={} trigger_type={} sent={}\n'.format(query, str(sent[0]), sent[1], sent_str)) 

            f_out.write('\n')

if __name__ == "__main__":
    main()


