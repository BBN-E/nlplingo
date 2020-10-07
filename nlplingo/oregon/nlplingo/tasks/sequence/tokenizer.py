from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
import torch


class Tokenizer(object):
    def __init__(self, opt):
        tokenizer_args = {'do_lower_case': False}
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', cache_dir=opt['cache_dir'],
                                                       **tokenizer_args)
        self.cls_token = self.tokenizer.cls_token  # e.g. <s>
        self.sep_token = self.tokenizer.sep_token  # e.g. </s>
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.mask_padding_with_zero = True

    def tokenize_to_ids(self, words, labels):
        """ The input 'labels' is already converted into ids via label_map
        """
        tokens = []
        first_subword_indices = []
        label_ids = []

        # xlmr has cls_token on left
        tokens = [self.cls_token] + tokens
        label_ids = [self.pad_token_label_id] + label_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        for word, label in zip(words, labels):
            subwords = self.tokenizer.tokenize(word)
            subword_ids = self.tokenizer.convert_tokens_to_ids(subwords)
            decoded_subwords = [self.tokenizer.decode([subword_id]) for subword_id in subword_ids]
            #print('subwords=', subwords, 'decoded_subwords=', decoded_subwords)
 
            i = 0
            while i < len(decoded_subwords) and decoded_subwords[i] == '':
                i += 1

            first_subword_indices.append(len(tokens) + i)
            #word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(subwords)
            input_ids.extend(subword_ids)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label] + [self.pad_token_label_id] * (len(subwords) - 1))
        #sys.exit(0)
        # add separator
        tokens += [self.sep_token]
        label_ids += [self.pad_token_label_id]
        input_ids += self.tokenizer.convert_tokens_to_ids([self.sep_token])
        # ==== Convert the tokens thus far, to their IDs. We will not be using `tokens` from here on
        #input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ##print('len(tokens)=', len(tokens), 'tokens=', tokens)
        ##print('len(input_ids)=', len(input_ids), 'input_ids=', input_ids)
        ##print('len(first_subword_indices)=', len(first_subword_indices), 'first_subword_indices=', first_subword_indices)
        #print('decoded=', [self.tokenizer.decode([input_id]) for input_id in input_ids])
        #print('decoded=', self.tokenizer.decode(input_ids))

        # ==== The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        assert len(input_ids) == len(input_mask) == len(label_ids)

        return torch.tensor(input_ids), input_mask, label_ids, first_subword_indices, tokens

    def do_padding(self, input_ids, input_mask, label_ids, max_seq_length):
        padding_length = max_seq_length - len(input_ids)

        input_ids += [self.pad_token] * padding_length
        input_mask += [0 if self.mask_padding_with_zero else 1] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        assert len(input_ids) == len(input_mask) == len(label_ids) == max_seq_length

        return input_ids, input_mask, label_ids
