import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import BaseEncoder

class CNNEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=230, 
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False,
                 is_embedding_vector=False,
                 features=None):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy; none means contextualized embeddings are used
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(CNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity,
                                         is_embedding_vector=is_embedding_vector, features=features)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, args):
        """
        Args:
            embedding_rep: (B, L), token indices if is_embedding_vector == False, (B, L, EMBED) if is_embedding_vector == True
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """

        if 'sentence_word_embedding_vector' in self.features.feature_order:
            embedding_rep_idx = self.features.feature_order['sentence_word_embedding_vector']
        else:
            # TODO: implement static embedding tokens
            pass

        pos1_idx = self.features.feature_order['arg1_position_data']
        pos2_idx = self.features.feature_order['arg2_position_data']

        embedding_rep = args[embedding_rep_idx]
        pos1 = args[pos1_idx]
        pos2 = args[pos2_idx]

        if not self.is_embedding_vector:
            # Check size of tensors
            if len(embedding_rep.size()) != 2 or embedding_rep.size() != pos1.size() or embedding_rep.size() != pos2.size():
                raise Exception("Size of token, pos1 ans pos2 should be (B, L)")

            x = torch.cat([self.word_embedding(embedding_rep),
                           self.pos1_embedding(pos1),
                           self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
            x = x.transpose(1, 2) # (B, EMBED, L)
            x = self.act(self.conv(x)) # (B, H, L)
            x = self.pool(x).squeeze(-1)
            x = self.drop(x)
        else:
            x = torch.cat([embedding_rep,
                           self.pos1_embedding(pos1),
                           self.pos2_embedding(pos2)], 2)  # (B, L, EMBED)
            x = x.transpose(1, 2)  # (B, EMBED, L)
            x = self.act(self.conv(x))  # (B, H, L)
            x = self.pool(x).squeeze(-1)
            x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class CNNEncoderLegacy(BaseEncoder):

    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(CNNEncoderLegacy, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        x = torch.cat([self.word_embedding(token),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        x = x.transpose(1, 2) # (B, EMBED, L)
        x = self.act(self.conv(x)) # (B, H, L)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)

class CNNEncoderLegacyDecode(CNNEncoderLegacy):
    def forward(self, token, pos1, pos2):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        length = token.size(1) # L
        word_embeddings = self.word_embedding(token) # of size B, L, EMBED
        word_embedding_dim = word_embeddings.size(2)
        total = sum([len(head_array) for head_array in pos1])
        result_tensor = torch.zeros((total, self.hidden_size))
        # pos1 is of size B x V x L where V is variable
        ct = 0
        # print('total', total) #  word_embeddings.shape)
        # print('word_shape', word_embeddings.shape)
        for i in range(word_embeddings.size(0)):
            v_len = len(pos1[i]) # V
            repeated_sentence_embedding = word_embeddings[i].unsqueeze(0).expand(v_len, length, word_embedding_dim) # V x L
            curr_head_array = torch.tensor(pos1[i]).long()
            curr_tail_array = torch.tensor(pos2[i]).long()
            x = torch.cat([repeated_sentence_embedding,
                           self.pos1_embedding(curr_head_array),
                           self.pos2_embedding(curr_tail_array)], 2)  # (V, L, EMBED)
            x = x.transpose(1, 2)  # (B, EMBED, L)
            x = self.act(self.conv(x))  # (B, H, L)
            maxpool = nn.MaxPool1d(length)
            x = maxpool(x).squeeze(-1)
            x = self.drop(x) # V x H
            result_tensor[ct:ct + v_len, :] = x
            ct += v_len
        result_tensor = result_tensor.to(self.device)
        return result_tensor

    def tokenize(self, item):
        return super().tokenize(item)