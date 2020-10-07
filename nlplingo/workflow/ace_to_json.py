import os
import argparse
import json

from nlplingo.annotation.ingestion import prepare_docs


def ace_to_json(params):
    word_embeddings = dict()
    train_docs = prepare_docs(params['data']['train']['filelist'], word_embeddings, params)
    dev_docs = prepare_docs(params['data']['dev']['filelist'], word_embeddings, params)
    test_docs = prepare_docs(params['data']['test']['filelist'], word_embeddings, params)

    for doc in train_docs:
        outpath = os.path.join(params['output_train'], doc.docid+'.json')
        with open(outpath, 'w', encoding="utf-8") as outfile:
            json.dump(doc.to_json(), outfile, indent=2, ensure_ascii=False)

    for doc in dev_docs:
        outpath = os.path.join(params['output_dev'], doc.docid+'.json')
        with open(outpath, 'w', encoding="utf-8") as outfile:
            json.dump(doc.to_json(), outfile, indent=2, ensure_ascii=False)

    for doc in test_docs:
        outpath = os.path.join(params['output_test'], doc.docid+'.json')
        with open(outpath, 'w', encoding="utf-8") as outfile:
            json.dump(doc.to_json(), outfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # ==== command line arguments, and loading of input parameter files ====
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--params', required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print(json.dumps(params, sort_keys=True, indent=4))

    if args.mode == 'ace_to_json':
        ace_to_json(params)
    else:
        raise RuntimeError('mode: {} is not implemented!'.format(args.mode))