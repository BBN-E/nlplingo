import torch
from torch import nn, optim
from .base_model import SentenceRE


class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(args)  # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)  # (B, N)
        return logits

# difference from above lies in the forward function
class SoftmaxNNLegacy(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, hidden_factor_size=1):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size * hidden_factor_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            # torch.set_num_threads(1)

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args)  # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)  # (B, N)
        return logits

# difference from above lies in the forward function
class SoftmaxNNLegacyMentionPool(SoftmaxNNLegacy):
    """
    Softmax classifier for sentence-level relation extraction.
    """
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        x, head_arrays, tail_arrays = self.sentence_encoder(*args)  # (B, H)
        #print('x', x.shape)
        # head_arrays = args[1]
        #print('len(head_arrays)', len(head_arrays))
        #print('len(head_arrays[0])', len(head_arrays[0]))
        #print('len(head_arrays[0])', len(head_arrays[1]))
        # tail_arrays = args[2]
        pooled_vector_dict = {}
        total = sum([len(head_array) for head_array in head_arrays])
        #print('total', total)
        result_tensor = torch.zeros((total, 2*768)) # TODO: hidden size is currently hard-coded; this should be fixed
        ct = 0
        for i in range(x.size(0)):
            curr_head_array = head_arrays[i]
            curr_tail_array = tail_arrays[i]
            for curr_head, curr_tail in zip(curr_head_array, curr_tail_array):
                head_start = curr_head[0]
                tail_start = curr_tail[0]
                head_end = curr_head[1]
                tail_end = curr_tail[1]
                head_key =  str(i) + '#' + str(head_start) + '#' + str(head_end)
                tail_key =  str(i) + '#' + str(tail_start) + '#' + str(tail_end)
                if head_key not in pooled_vector_dict:
                    head_vectors = torch.narrow(x[i], 0, int(head_start), int(head_end - head_start))  # HEAD_LENGTH X EMBED
                    head_max = torch.max(head_vectors, 0)[0]
                    pooled_vector_dict[head_key] = head_max

                if tail_key not in pooled_vector_dict:
                    tail_vectors = torch.narrow(x[i], 0, int(tail_start), int(tail_end - tail_start))  # TAIL_LENGTH X EMBED
                    tail_max = torch.max(tail_vectors, 0)[0]
                    pooled_vector_dict[tail_key] = tail_max

                #print('head shape', head_max.shape)
                result = (torch.cat((pooled_vector_dict[head_key], pooled_vector_dict[tail_key]))).unsqueeze(0) #.to(self.device)
                #print('result shape', result.shape)
                result_tensor[ct, :] = result
                ct += 1
        result_tensor = result_tensor.to(self.device)

        rep = self.drop(result_tensor)
        logits = self.fc(rep)  # (B, N)
        return logits