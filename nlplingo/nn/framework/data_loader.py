import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
from nlplingo.common.scoring import eer_score_breakdown, arg_score_breakdown, entity_relation_score_breakdown
import nlplingo

class SentenceREDecodeDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, triplets, rel2id, tokenizer, optimize_batches, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.data = triplets
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.optimize_batches = optimize_batches
        self.kwargs = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if not self.optimize_batches:
            seq = list(self.tokenizer(item, **self.kwargs))
        else:
            seq = item
        return seq  # seq1, seq2, ...

    def collate_fn(data):
        data = list(zip(*data))
        seqs = data
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return batch_seqs

    def collate_fn_optimized(data):
        global_max_sent_length = nlplingo.nn.constants.get_global_max_sent_length()
        max_sent_length = min(global_max_sent_length, data[-1][0]) # global_max_sent_length # data[-1][0] # global_max_sent_length #
        # print('max_sent_length', max_sent_length)
        result = []
        for data_piece in data:
            interm = data_piece[1]
            avai_len = len(interm)
            while len(interm) < max_sent_length:
                interm.append(0)  # 0 is id for [PAD]
            interm = interm[:max_sent_length]
            interm = torch.tensor(interm).long().unsqueeze(0)  # (1, L)
            att_mask = torch.zeros(interm.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            result.append((interm, att_mask))

        final = list(zip(*result))
        seqs = final
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return batch_seqs

    def collate_fn_mention_pool(data):
        """
        This function was used during decoding for an earlier (now defunct) mention pool implementation
        that did not attach multiple relations to a unique sentence.
        :return:
        """
        global_max_sent_length = nlplingo.nn.constants.get_global_max_sent_length()
        max_sent_length = min(global_max_sent_length, data[-1][0]) # global_max_sent_length # data[-1][0] # global_max_sent_length #
        # print('max_sent_length', max_sent_length)
        result = []
        for data_piece in data:
            interm = data_piece[1]
            avai_len = len(interm)
            while len(interm) < max_sent_length:
                interm.append(0)  # 0 is id for [PAD]
            interm = interm[:max_sent_length]
            interm = torch.tensor(interm).long().unsqueeze(0)  # (1, L)
            att_mask = torch.zeros(interm.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            head_indices = torch.tensor(data_piece[2]).long().unsqueeze(0)
            tail_indices = torch.tensor(data_piece[3]).long().unsqueeze(0)
            result.append((interm, att_mask, head_indices, tail_indices))

        final = list(zip(*result))
        seqs = final
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return batch_seqs

    def collate_fn_mention_pool_decode(data):
        global_max_sent_length = nlplingo.nn.constants.get_global_max_sent_length()
        max_sent_length = min(global_max_sent_length, data[-1][0])
        result = []
        for data_piece in data:
            interm = data_piece[1]
            avai_len = len(interm)
            while len(interm) < max_sent_length:
                interm.append(0)  # 0 is id for [PAD]
            interm = interm[:max_sent_length]
            interm = torch.tensor(interm).long().unsqueeze(0)  # (1, L)
            att_mask = torch.zeros(interm.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            result.append((interm, att_mask, data_piece[2], data_piece[3]))

        final = list(zip(*result))
        seqs = final
        return torch.cat(seqs[0],0), torch.cat(seqs[1], 0), seqs[2], seqs[3]

    def collate_fn_cnn_decode(data):
        global_max_sent_length = nlplingo.nn.constants.get_global_max_sent_length()
        global_blank_id = nlplingo.nn.constants.get_global_blank_id() # move this to kwargs?
        max_sent_length = min(global_max_sent_length, data[-1][0])
        result = []
        for data_piece in data:
            interm = data_piece[1]
            interm_pos1 = []
            interm_pos2 = []
            while len(interm) < max_sent_length:
               interm.append(global_blank_id)

            # pad everything properly in this batch
            for pos1 in data_piece[2]:
                while len(pos1) < max_sent_length:
                    pos1.append(0)
                interm_pos1.append(pos1[:max_sent_length])

            for pos2 in data_piece[3]:
                while len(pos2) < max_sent_length:
                    pos2.append(0)
                interm_pos2.append(pos2[:max_sent_length])

            interm = interm[:max_sent_length]
            interm = torch.tensor(interm).long().unsqueeze(0)  # (1, L)
            result.append((interm, interm_pos1, interm_pos2))

        final = list(zip(*result))
        seqs = final
        return torch.cat(seqs[0],0), seqs[1], seqs[2]

# True means decode_mode is turned on
collate_table = {
    True : {'bert_mention' : SentenceREDecodeDataset.collate_fn_mention_pool_decode, 'cnn' : SentenceREDecodeDataset.collate_fn,
             'cnn_opt' : SentenceREDecodeDataset.collate_fn_cnn_decode},
    False : {'bert_mention' : SentenceREDecodeDataset.collate_fn_mention_pool} # TODO: get rid of this case
}

def SentenceREDecodeLoader(hyper_params, triplets, rel2id, tokenizer, batch_size,
                           shuffle, num_workers=0, optimize_batches=False, **kwargs):
        dataset = SentenceREDecodeDataset(triplets, rel2id=rel2id, tokenizer=tokenizer, optimize_batches=optimize_batches, kwargs=kwargs)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=collate_table[hyper_params.decode_mode][hyper_params.encoder])
        return data_loader

class SentenceRETrainDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, path, rel2id, tokenizer, event_domain, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.event_domain = event_domain
        self.data_labels = []

        # Load the file
        f = open(path)
        self.data = []
        ct = 0
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                obj = eval(line)
                self.data.append(obj)
                self.data_labels.append(self.rel2id[obj['relation']])
        f.close()
        logging.info("Loaded sentence RE dataset {} with {} lines and {} relations.".format(path, len(self.data),
                                                                                            len(self.rel2id)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        #print('item', item)
        #print('index', index)
        assert(item is not None)
        seq = list(self.tokenizer(item, **self.kwargs))
        return [self.rel2id[item['relation']]] + seq  # label, seq1, seq2, ...

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + batch_seqs

    """
    def __getitem__(self, index):
        item = self.data[index]
        #print('item', item)
        #print('index', index)
        assert(item is not None)
        vals = self.tokenizer(item, **self.kwargs)
        if vals is None:
            return (False, [])
        else:
            seq = list(vals)
        return (True, [self.rel2id[item['relation']]] + seq)  # label, seq1, seq2, ...

    def collate_fn(data):
        data_items = data # list(*data)

        final_data = []
        for idx, item in enumerate(data_items):
            if item[0]:
                final_data.append(item[1])

        data = list(zip(*final_data))
        labels = data[0]
        batch_labels = torch.tensor(labels).long()  # (B)
        seqs = data[1:]
        batch_seqs = []
        for idx, seq in enumerate(seqs):
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + batch_seqs
    """

    def eval(self, pred_result, model_name, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """


        neg = -1

        if model_name.startswith('event-event-relation'):
            for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
                if name in self.event_domain.eer_types:
                    if use_name:
                        neg = name
                    else:
                        neg = self.event_domain.eer_types[name]
                    break

            result = eer_score_breakdown(self.event_domain, self.data_labels, pred_result, neg, unfold_tuple=False)
            logging.info('Evaluation result: {}.'.format(result))
        elif model_name.startswith('event-argument'):
            for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
                if name in self.event_domain.event_roles:
                    if use_name:
                        neg = name
                    else:
                        neg = self.event_domain.event_roles[name]
                    break

            result = arg_score_breakdown(self.event_domain, self.data_labels, pred_result, neg, unfold_tuple=False)
            logging.info('Evaluation result: {}.'.format(result))
        elif model_name.startswith('entity-entity'):
            for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
                if name in self.event_domain.entity_relation_types:
                    if use_name:
                        neg = name
                    else:
                        neg = self.event_domain.entity_relation_types[name]
                    break

            result = entity_relation_score_breakdown(self.event_domain, self.data_labels, pred_result, neg, unfold_tuple=False)
            logging.info('Evaluation result: {}.'.format(result))
        else:
            raise Exception('Model type {} not supported'.format(model_name))
        return result


def SentenceRETrainLoader(path, rel2id, tokenizer, batch_size,
                          shuffle, event_domain=None, num_workers=8, collate_fn=SentenceRETrainDataset.collate_fn, **kwargs):
    dataset = SentenceRETrainDataset(path=path, rel2id=rel2id, tokenizer=tokenizer, event_domain=event_domain, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
