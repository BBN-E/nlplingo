import logging
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AlbertConfig, AlbertModel, AlbertTokenizer, DistilBertConfig, DistilBertModel, DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer
import numpy as np

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False, encoder="bert_original", is_decode=True):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        self.encoder = encoder
        self.is_decode = is_decode
        logging.info('Loading BERT pre-trained checkpoint.')
        config_file = pretrain_path + '/config.json'
        print('encoder', encoder)
        if encoder == "bert_original" or encoder == "bert_legacy" or encoder == "bert_mention":
            if self.is_decode:
                config = BertConfig.from_pretrained(config_file)
                self.bert = BertModel(config)
            else:
                self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
            self.tokenizer_optimized = BertWordPieceTokenizer(pretrain_path + "/vocab.txt", lowercase=True)
        elif encoder == "albert_original" or encoder == "albert_mention":
            print('Loading Albert model...')
            if self.is_decode:
                config = AlbertConfig.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/albert/config.json')
                self.bert = AlbertModel(config)
            else:
                self.bert = AlbertModel.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/albert/')

            self.tokenizer = AlbertTokenizer.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/albert/')
        elif encoder == "distilbert_original":
            if self.is_decode:
                config = DistilBertConfig.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/distilbert/config.json')
                self.bert = DistilBertModel(config)
            else:
                self.bert = DistilBertModel.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/distilbert/')
            self.tokenizer = DistilBertTokenizer.from_pretrained('/nfs/raid88/u10/users/jcai/nlplingo_trained_models/transformers/distilbert/')
        else:
            raise Exception('Encoder not supported.')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        if self.encoder != 'bert_legacy':
            if self.encoder == 'bert_original' or self.encoder == 'albert_original':
                x, _ = self.bert(token, attention_mask=att_mask)
            else:
                x = self.bert(token, attention_mask=att_mask)[0]
            # print('shape', x.shape)
            return torch.mean(x, 1)
        else:
            _, x = self.bert(token, attention_mask=att_mask)
            return x

    def tokenize(self, item, blank_padding=True, mention_pool=False):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            if self.mask_entity:
                ent0 = ['[unused4]']
                ent1 = ['[unused5]']
                if rev:
                    ent0 = ['[unused5]']
                    ent1 = ['[unused4]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence

        head_indices = []
        tail_indices = []
        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        for token in tokens:
            token = token.lower()
            if not mention_pool:
                if cur_pos == pos_head[0] and not self.mask_entity:
                    re_tokens.append('[unused0]')
                if cur_pos == pos_tail[0] and not self.mask_entity:
                    re_tokens.append('[unused1]')

            if mention_pool:
                if cur_pos == pos_head[0]:
                    head_indices.append(len(re_tokens))
                if cur_pos == pos_tail[0]:
                    tail_indices.append(len(re_tokens))

            if is_token:
                re_tokens += self.tokenizer.tokenize(token)
            else:
                re_tokens.append(token)

            if not mention_pool:
                if cur_pos == pos_head[1] - 1 and not self.mask_entity:
                    re_tokens.append('[unused2]')

                if cur_pos == pos_tail[1] - 1 and not self.mask_entity:
                    re_tokens.append('[unused3]')

            if mention_pool:
                if cur_pos == pos_head[1] - 1:
                    head_indices.append(len(re_tokens))
                if cur_pos == pos_tail[1] - 1:
                    tail_indices.append(len(re_tokens))
            cur_pos += 1
        re_tokens.append('[SEP]')
        #print('head_indices', head_indices)
        #print('tail_indices', tail_indices)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        if mention_pool:
            assert(len(head_indices) == 2)
            assert(len(tail_indices) == 2)
            # if invalid, we return None

            if (head_indices[1] > self.max_length) or (tail_indices[1] > self.max_length):
                if self.is_decode:
                    return None
                else:
                    raise Exception('The head/tail indices should be valid.')        # Padding
                
        if blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1

            if not mention_pool:
                return indexed_tokens, att_mask
            else:
                head_indices = torch.tensor(head_indices).long().unsqueeze(0)
                tail_indices = torch.tensor(tail_indices).long().unsqueeze(0)
                return indexed_tokens, att_mask, head_indices, tail_indices
        else:
            if not mention_pool:
                return (avai_len, indexed_tokens)
            else:
                return (avai_len, indexed_tokens, head_indices, tail_indices)

    def tokenize_mention_pool_decode(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        sentence = item[0]
        is_token = False
        assert(len(item[1]) == len(item[2]))
        head_indices = item[1]
        tail_indices = item[2]

        def find_intersection(intervals, query):
            """Find intersections between intervals.
            Intervals are open and are represented as pairs (lower bound,
            upper bound).

            Arguments:
            intervals: array_like, shape=(N, 2) -- Array of intervals.
            query: array_like, shape=(2,) -- Interval to query.

            Returns:
            Array of indexes of intervals that overlap with query.

            """
            endpoints = np.reshape(intervals, -1)
            lower, upper = query
            i = np.searchsorted(endpoints, lower, side='right')
            j = np.searchsorted(endpoints, upper, side='left')
            s = i // 2
            e = (j+ 1) // 2
            # assert(e > s)
            return (s + 1, e + 1), e > s

        sent0 = self.tokenizer_optimized.encode(sentence) #, add_special_tokens=False) # return_offsets_mapping=True)
        #sent1 = self.tokenizer.encode(sentence) #, add_special_tokens=False)
        #assert(sent0.ids == sent1)
        indexed_tokens = sent0.ids
        offsets = [(k[0], k[1] - 1) for k in sent0.offsets[1:-1]]
        head_arrays = []
        tail_arrays = []
        del_indices = set()
        ct = 0
        for head_interval, tail_interval in zip(head_indices, tail_indices):
            h, exists_head = find_intersection(offsets, head_interval)
            t, exists_tail = find_intersection(offsets, tail_interval)
            if (h[1] > self.max_length) or (t[1] > self.max_length) or (not(exists_head)) or (not(exists_tail)):
                del_indices.add(ct)
            else:
                head_arrays.append(h)
                tail_arrays.append(t)
            ct += 1
        #print('head_arrays', head_arrays)
        #print('tail_arrays', tail_arrays)
        avai_len = len(indexed_tokens)
        return (avai_len, indexed_tokens, head_arrays, tail_arrays), del_indices

class BERTEncoderMentionPool(BERTEncoder):
    def forward(self, token, att_mask, head_indices, tail_indices):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            head_indices: (B, 2), head start and end indices
            tail_indices: (B, 2), tail start and end indices
        Return:
            (B, 2*H), representations for sentences
        """
        x, _ = self.bert(token, attention_mask=att_mask)

        result_tensor = torch.zeros((x.size(0), 2*self.hidden_size))
        for i in range(x.size(0)):
            head_start = head_indices[i][0].item()
            head_end = head_indices[i][1].item()
            tail_start = tail_indices[i][0].item()
            tail_end = tail_indices[i][1].item()
            if head_end >= self.max_length or tail_end >= self.max_length:
                raise Exception('should never happen')
                #result_tensor[i, :]
                #result_lst.append(torch.zeros((1, self.hidden_size*2)))
                continue
            #print('x shape', x[i].shape)
            #print('head_end-head_start', head_end - head_start)
            #print('head_start', head_start)
            #print('head_start', type(head_start))
            head_vectors = torch.narrow(x[i], 0, head_start, head_end - head_start) # HEAD_LENGTH X EMBED
            tail_vectors = torch.narrow(x[i], 0, tail_start, tail_end - tail_start) # TAIL_LENGTH X EMBED
            head_max = torch.max(head_vectors, 0)[0]
            tail_max = torch.max(tail_vectors, 0)[0]
            #print('head shape', head_max.shape)
            result = (torch.cat((head_max, tail_max))).unsqueeze(0) #.to(self.device)
            #print('result shape', result.shape)
            result_tensor[i, :] = result
            # result_lst.append(result) # appends a tensor of size (1, 2*EMBED) to result_lst
        # result_lst.to(self.device)
        # final_result = torch.cat(result_lst, 0)
        return result_tensor.to(self.device)

class BERTEncoderMentionPoolDecode(BERTEncoder):
    def forward(self, token, att_mask, head_array, tail_array):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            head_indices: (B, 2), head start and end indices
            tail_indices: (B, 2), tail start and end indices
        Return:
            (B, 2*H), representations for sentences
        """
        x, _ = self.bert(token, attention_mask=att_mask)
        return x, head_array, tail_array

# TODO: figure out a way to unify previous and current class
# probably use an is_train option
class BERTEncoderTrain(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False, encoder="bert"):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        if encoder == "bert":
            self.bert = BertModel.from_pretrained(pretrain_path)
            self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        elif encoder == "albert":
            self.bert = AlbertModel.from_pretrained(pretrain_path)
            self.tokenizer = AlbertTokenizer.from_pretrained(pretrain_path)
        elif encoder == "distilbert":
            self.bert = DistilBertModel.from_pretrained('/nfs/raid88/u10/users/jcai/code/download-hugging/transformers/distilbert')
            self.tokenizer = DistilBertTokenizer.from_pretrained('/nfs/raid88/u10/users/jcai/code/download-hugging/transformers/distilbert')
        else:
            raise Exception('Encoder not supported.')

    def forward(self, args):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(args[0], attention_mask=args[1])
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            if self.mask_entity:
                ent0 = ['[unused4]']
                ent1 = ['[unused5]']
                if rev:
                    ent0 = ['[unused5]']
                    ent1 = ['[unused4]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence

        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0] and not self.mask_entity:
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0] and not self.mask_entity:
                re_tokens.append('[unused1]')
            if is_token:
                re_tokens += self.tokenizer.tokenize(token)
            else:
                re_tokens.append(token)
            if cur_pos == pos_head[1] - 1 and not self.mask_entity:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[1] - 1 and not self.mask_entity:
                re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask

class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768 * 2
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        hidden, _ = self.bert(token, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            if self.mask_entity:
                ent0 = ['[unused4]']
                ent1 = ['[unused5]']
                if rev:
                    ent0 = ['[unused5]']
                    ent1 = ['[unused4]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence

        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = 0
        pos2 = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[unused1]')
            if is_token:
                re_tokens += self.tokenizer.tokenize(token)
            else:
                re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(
            0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2
