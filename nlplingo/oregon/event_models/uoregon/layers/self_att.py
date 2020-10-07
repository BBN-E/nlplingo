import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from nlplingo.oregon.event_models.uoregon.tools.global_constants import *

# n_position = MAX_BERT_TOKENS - 50 = 512 - 50 ; is this sequence length?
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table.'''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(input_masks):
    assert input_masks.dim() == 2
    return input_masks.ne(PAD_ID).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(input_masks):
    ''' For masking out padding tokens in each example, repeated for each query running over the example. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = input_masks.size(1)
    padding_mask = input_masks.eq(PAD_ID)  # [batch_size, num_key_tokens]
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # batch_size x num_q_tokens x num_key_tokens

    return padding_mask


def get_position_ids(input_masks, device):
    with torch.no_grad():
        batch_size = input_masks.size(0)
        seq_len = input_masks.size(1)
        position_ids = torch.stack([torch.arange(1, seq_len + 1) for _ in range(batch_size)],
                                   dim=0)  # [batch size, seq len]
        return position_ids.to(device) * input_masks.bool().long().to(device)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)        # every slice along dim=2 will sum to 1.0

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # [batch size, num q tokens, input_dim]
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, input_dim, opt):
        super().__init__()

        self.n_head = opt['self_att_heads']     # 1
        self.d_qkv = opt['self_att_d_qkv']      # 200

        self.w_qs = nn.Linear(input_dim, self.n_head * self.d_qkv)  # queries
        self.w_ks = nn.Linear(input_dim, self.n_head * self.d_qkv)  # keys
        self.w_vs = nn.Linear(input_dim, self.n_head * self.d_qkv)  # values

        # TODO: is this Xavier or Kaiming? You have denominator as fan-in + fan-out, but yet you have 2.0 as numerator
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (input_dim + self.d_qkv)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (input_dim + self.d_qkv)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (input_dim + self.d_qkv)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_qkv, 0.5))
        self.layer_norm = nn.LayerNorm(input_dim)   # normalize over last dimension which is expected to be of size 'input_dim'

        self.fc = nn.Linear(self.n_head * self.d_qkv, input_dim)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(opt['self_att_dropout'])  # 0.1

    def forward(self, q, k, v, mask):
        d_k, d_v, n_head = self.d_qkv, self.d_qkv, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q  # [batch size, num q tokens, input_dim]

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    d_k)  # [n_heads x batch_size, num q_tokens, dk]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [n_heads x batch_size, num key tokens, dk]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [n_heads x batch_size, num key tokens, dv]

        mask = mask.repeat(n_head, 1, 1)  # [n_heads x batch_size, num q_tokens, num key tokens]
        output, attn = self.attention(q, k, v, mask=mask)  # [n_heads x batch size, num q tokens, dv]

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q,
                                                              -1)  # [batch size, num q tokens, n_heads x dv]

        output = self.dropout(self.fc(output))  # [batch size, num q tokens, input_dim]
        output = self.layer_norm(output + residual)

        return output, attn


class SelfAttention_Layer(nn.Module):
    def __init__(self, input_dim, opt):
        super(SelfAttention_Layer, self).__init__()
        self.slf_attn = MultiHeadAttention(input_dim, opt)
        self.pos_ffn = PositionwiseFeedForward(input_dim, input_dim * 4, dropout=opt['self_att_dropout'])

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        ''' non_pad_mask: zeros at padding tokens, slf_attn_mask: zeros at real tokens, ones at padding tokens to use masked_fill'''
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class SelfAttention(nn.Module):
    '''
    Paper's setting:
    num layers: N = 6
    num multi-head attentions: h = 8
    d_model = d_qkv * h = 512 => d_qkv = 64
    hidden dim of position embedding: 2048
    adam optimizer, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9, warmup_steps = 4000
    lrate = 1/sqrt(d_model) * min(1/sqrt(step_num), step_num * 4000^(-1.5))
    '''

    def __init__(self, input_dim, opt):
        super(SelfAttention, self).__init__()
        #print('========= self_att.SelfAttention.__init__ START ===========')
        """ decode.bash
        position_embed_for_satt= 1
        self_att_layers= 6
        """
        self.opt = opt
        #print('position_embed_for_satt=', self.opt['position_embed_for_satt'])
        #print('self_att_layers=', self.opt['self_att_layers'])
        self.attention_layers = nn.ModuleList([
            SelfAttention_Layer(input_dim, opt)
            for _ in range(self.opt['self_att_layers'])])       # 6
        if self.opt['position_embed_for_satt']:                 # 1
            self.position_embedding = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(SAFE_BERT_TOKENS, input_dim, padding_idx=PAD_ID),
                freeze=True)
        #print('========= self_att.SelfAttention.__init__ END ===========')

    def forward(self, word_reps, pad_masks):
        #print('========= self_att.SelfAttention.forward START ===========')
        input_masks = pad_masks.long().eq(0).bool()
        #print('input_masks.shape=', input_masks.shape)
        # to remove padding tokens, zeros at real tokens,
        # ones at padding tokens to use masked_fill
        # slf_attn_mask.shape = [batch size, seq len, seq len]
        slf_attn_mask = get_attn_key_pad_mask(input_masks)
        #print('slf_attn_mask.shape=', slf_attn_mask.shape)
        # to keep real tokens,
        # zeros at padding tokens to directly multiply this mask to tokens' representations
        # non_pad_mask.shape = [batch size, seq len, 1]
        non_pad_mask = get_non_pad_mask(input_masks)
        #print('non_pad_mask.shape=', non_pad_mask.shape)
        enc_output = word_reps
        #print('enc_output.shape=', enc_output.shape)
        #print('position_embed_for_satt=', self.opt['position_embed_for_satt'])

        if self.opt['position_embed_for_satt']:
            position_ids = get_position_ids(input_masks, self.opt['device'])
            #print('position_ids.shape=', position_ids.shape)
            enc_output += self.position_embedding(position_ids)
            #print('enc_output.shape=', enc_output.shape)

        output_reps = []
        for enc_layer in self.attention_layers:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            output_reps.append(enc_output)
        #print('========= self_att.SelfAttention.forward END ===========')

        return output_reps[-1], enc_slf_attn
