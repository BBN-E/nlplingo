import torch.nn as nn
import torch


class CRF(nn.Module):
    def __init__(self, tag_map, opt):
        super().__init__()
        self.opt = opt
        self.tag_map = tag_map
        self.num_tags = len(tag_map)
        # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        self.trans.data[tag_map["<SOS>"], :] = -10000  # no transition to SOS
        self.trans.data[:, tag_map["<EOS>"]] = -10000  # no transition from EOS except to PAD
        self.trans.data[:, tag_map["<PAD>"]] = -10000  # no transition from PAD except to PAD
        self.trans.data[tag_map["<PAD>"], :] = -10000  # no transition to PAD except from EOS
        self.trans.data[tag_map["<PAD>"], tag_map["<EOS>"]] = 0
        self.trans.data[tag_map["<PAD>"], tag_map["<PAD>"]] = 0

    def forward(self, h, mask):  # forward algorithm
        # initialize forward variables in log space
        batch_size, _, _ = h.shape
        score = torch.Tensor(batch_size, self.num_tags).fill_(-10000).to(self.opt['device'])  # [B, C]
        score[:, self.tag_map["<SOS>"]] = 0.
        trans = self.trans.unsqueeze(0)  # [1, C, C]
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[self.tag_map["<EOS>"]])
        return score  # partition function

    def score(self, h, y0, mask):  # calculate the score of a given sequence
        batch_size, _, _ = h.shape
        score = torch.Tensor(batch_size).fill_(0.).to(self.opt['device'])
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
            trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
            score += (emit_t + trans_t) * mask_t
        last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[self.tag_map["<EOS>"], last_tag]
        return score

    def decode(self, h, mask):  # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        batch_size, seq_len, _ = h.shape
        bptr = torch.Tensor().long().to(self.opt['device'])
        score = torch.Tensor(batch_size, self.num_tags).fill_(-10000).to(self.opt['device'])
        score[:, self.tag_map["<SOS>"]] = 0.

        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans  # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2)  # best previous scores and tags
            score_t += h[:, t]  # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.tag_map["<EOS>"]]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size):
            i = best_tag[b]  # best tag
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()
        padded_path = []
        for b in range(batch_size):
            padded_path.append(
                best_path[b] + [self.tag_map["<PAD>"]] * (seq_len - len(best_path[b]))
            )
        padded_path = torch.Tensor(padded_path).long().to(self.opt['device'])
        return padded_path


def preds_to_tags(batch_preds, mask, tag_map):
    '''
    preds.shape = batch size, seq len
    '''
    inverse_map = dict([(v, k) for k, v in tag_map.items()])
    lengths = torch.sum(mask, dim=-1).data.cpu().numpy()
    batch_size, seq_len = batch_preds.shape
    batch_preds = batch_preds.data.cpu().numpy()
    tags = []
    for b_id in range(batch_size):
        preds = batch_preds[b_id][:lengths[b_id]]


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
