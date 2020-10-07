from nlplingo.oregon.event_models.uoregon.models.pipeline._01.modules.ED_model import EDModel
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.iterators import EDIterator
from nlplingo.oregon.event_models.uoregon.models.pipeline._01.local_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *
import torch
from nlplingo.oregon.event_models.uoregon.define_opt import logger
from datetime import datetime


class EDTrainer:
    def __init__(self, opt, eval_mode=False):
        self.opt = opt
        self.model = EDModel(opt)
        self.model.to(self.opt['device'])

        if not self.opt['finetune_xlmr']:                           # skip
            for name, param in self.model.named_parameters():
                if 'xlmr' in name:
                    param.requires_grad = False
        elif self.opt['finetune_on_arb']:  # only update xlmr embeddings    # skip
            for name, param in self.model.named_parameters():
                found_in_list = False
                for finetuned_layer in opt['finetuned_xlmr_layers']:
                    if name.startswith(finetuned_layer):
                        found_in_list = True
                        break
                if not found_in_list:
                    param.requires_grad = False

        self._print_args()
        """
        > trainable params: ED model
        >>> upos_embedding.weight: torch.Size([19, 30])
        >>> xlmr_embedding.model.encoder.sentence_encoder.embed_tokens.weight: torch.Size([250002, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.embed_positions.weight: torch.Size([514, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.0.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.1.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.2.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.3.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.4.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.5.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.6.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.7.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.8.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.9.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.10.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.k_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.k_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.v_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.v_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.q_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.q_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.out_proj.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn.out_proj.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.self_attn_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.fc1.weight: torch.Size([3072, 768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.fc1.bias: torch.Size([3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.fc2.weight: torch.Size([768, 3072])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.fc2.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.final_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.layers.11.final_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.emb_layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.sentence_encoder.emb_layer_norm.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.lm_head.bias: torch.Size([250002])
        >>> xlmr_embedding.model.encoder.lm_head.dense.weight: torch.Size([768, 768])
        >>> xlmr_embedding.model.encoder.lm_head.dense.bias: torch.Size([768])
        >>> xlmr_embedding.model.encoder.lm_head.layer_norm.weight: torch.Size([768])
        >>> xlmr_embedding.model.encoder.lm_head.layer_norm.bias: torch.Size([768])
        >>> self_att.attention_layers.0.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.0.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.0.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.0.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.0.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.0.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.0.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.0.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.0.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.0.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.0.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.0.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.0.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.0.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.0.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.0.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.1.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.1.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.1.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.1.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.1.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.1.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.1.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.1.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.1.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.1.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.1.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.1.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.1.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.1.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.1.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.1.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.2.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.2.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.2.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.2.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.2.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.2.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.2.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.2.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.2.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.2.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.2.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.2.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.2.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.2.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.2.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.2.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.3.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.3.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.3.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.3.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.3.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.3.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.3.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.3.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.3.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.3.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.3.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.3.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.3.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.3.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.3.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.3.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.4.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.4.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.4.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.4.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.4.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.4.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.4.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.4.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.4.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.4.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.4.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.4.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.4.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.4.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.4.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.4.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.5.slf_attn.w_qs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.5.slf_attn.w_qs.bias: torch.Size([200])
        >>> self_att.attention_layers.5.slf_attn.w_ks.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.5.slf_attn.w_ks.bias: torch.Size([200])
        >>> self_att.attention_layers.5.slf_attn.w_vs.weight: torch.Size([200, 798])
        >>> self_att.attention_layers.5.slf_attn.w_vs.bias: torch.Size([200])
        >>> self_att.attention_layers.5.slf_attn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.5.slf_attn.layer_norm.bias: torch.Size([798])
        >>> self_att.attention_layers.5.slf_attn.fc.weight: torch.Size([798, 200])
        >>> self_att.attention_layers.5.slf_attn.fc.bias: torch.Size([798])
        >>> self_att.attention_layers.5.pos_ffn.w_1.weight: torch.Size([3192, 798, 1])
        >>> self_att.attention_layers.5.pos_ffn.w_1.bias: torch.Size([3192])
        >>> self_att.attention_layers.5.pos_ffn.w_2.weight: torch.Size([798, 3192, 1])
        >>> self_att.attention_layers.5.pos_ffn.w_2.bias: torch.Size([798])
        >>> self_att.attention_layers.5.pos_ffn.layer_norm.weight: torch.Size([798])
        >>> self_att.attention_layers.5.pos_ffn.layer_norm.bias: torch.Size([798])
        >>> gcn_layer.W.0.weight: torch.Size([798, 798])
        >>> gcn_layer.W.0.bias: torch.Size([798])
        >>> gcn_layer.W.1.weight: torch.Size([798, 798])
        >>> gcn_layer.W.1.bias: torch.Size([798])
        >>> biw2v_embedding.weight: torch.Size([354186, 300])
        >>> fc_ED.0.weight: torch.Size([200, 1896])
        >>> fc_ED.0.bias: torch.Size([200])
        >>> fc_ED.2.weight: torch.Size([16, 200])
        >>> fc_ED.2.bias: torch.Size([16])
        n_trainable_params: 420657080, n_nontrainable_params: 368676
        """

        if not eval_mode:
            self.train_iterator = EDIterator(
                xlmr_model=self.model.xlmr_embedding,
                # data_path=os.path.join(MODEL_DIR, 			# <==
                data_path=os.path.join(self.opt['datapoint_dir'],	# ==>
                                       opt['data_map']['ED']['train'].format(get_data_dir(self.opt['data'])))
            )
            print('ED model : Training data: {}'.format(self.train_iterator.num_examples))

            self.dev_iterator = EDIterator(
                xlmr_model=self.model.xlmr_embedding,
                # data_path=os.path.join(MODEL_DIR,			# <==
                data_path=os.path.join(self.opt['datapoint_dir'],	# ==>
                                       opt['data_map']['ED']['dev'].format(get_data_dir(self.opt['data']))),
                is_eval_data=True
            )
            print('ED model : Dev data: {}'.format(self.dev_iterator.num_examples))

        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, betas=(0.9, 0.98), lr=opt['lr'])

        self.best_result = {
            'epoch': 0,
            'en_p': 0,
            'en_r': 0,
            'en_f1': 0
        }

        cout = '*' * 100 + '\n'
        cout += 'Opt:\n'
        for arg in opt:
            if arg not in ['biw2v_vecs']:
                cout += '{}: {}\n'.format(arg, opt[arg])
        cout += '*' * 100 + '\n'
        print(cout)
        logger.info(cout)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()     # YS: number of parameters for this item
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('-' * 100)
        logger.info('> trainable params: ED model')
        print('> trainable params: ED model')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info('>>> {0}: {1}'.format(name, param.shape))
                print('>>> {0}: {1}'.format(name, param.shape))
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('-' * 100)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        for b_id, batch in enumerate(self.train_iterator):
            batch = move_to_device(batch, self.opt['device'])
            loss, probs, preds = self.model(batch)

            loss.backward()
            if self.opt['grad_clip_xlmr']:
                params_for_clip = self.model.parameters()
            else:
                params_for_clip = [param for name, param in self.model.named_parameters() if
                                   not name.startswith('xlmr_embedding')]
            torch.nn.utils.clip_grad_norm_(params_for_clip, self.opt['max_grad_norm'])

            self.optimizer.step()
            self.optimizer.zero_grad()
            ###################################
            if b_id % 20 == 0:
                print(
                    '{}: step {}/{} (epoch {}/{}), loss = {:.3f}'.format(
                        datetime.now(), b_id, self.train_iterator.__len__(), epoch, self.opt['num_epoch'],
                        loss.item()
                    ))
                logger.info(
                    '{}: step {}/{} (epoch {}/{}), loss = {:.3f}'.format(
                        datetime.now(), b_id, self.train_iterator.__len__(), epoch, self.opt['num_epoch'],
                        loss.item()
                    ))

        self.eval(epoch)
        if not self.opt['save_last_epoch'] and not self.opt['delete_nonbest_ckpts']:
            print('Saving model...')
            logger.info('Saving model...')
            self.save_model(epoch)
        self.train_iterator.shuffle_batches()
        print('=' * 50)
        logger.info('=' * 50)

    def prepare_training(self):
        if self.opt['train_strategy'].startswith('retrain'):
            # *********** initialize pretrained model ***********
            if self.opt['initialize_with_pretrained']:
                ED_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
                                   len(re.findall(
                                       r'ED[.]epoch[-]\d+[.]config',
                                       fname)) > 0]
                if len(ED_config_paths) > 0:
                    ED_config_path = ED_config_paths[0]
                    self.opt['ED_eval_epoch'] = int(
                        re.findall(r'epoch[-](\d+)[.]config',
                                   ED_config_path)[0])

                    print('Initializing model with pretrained model at checkpoint: {}'.format(
                        self.opt['ED_eval_epoch']))
                    logger.info('Initializing model with pretrained model at checkpoint: {}'.format(
                        self.opt['ED_eval_epoch']))
                    self.load_saved_model()
                    self.eval(self.opt['ED_eval_epoch'])
                else:
                    print('No pretrained model found!')
                    logger.info('No pretrained model found!')
            else:
                # *********** English checkpoints *************
                existing_ckpts = [fname for fname in get_files_in_dir('checkpoints') if
                                  os.path.basename(fname).startswith('ED')]
                print('Deleting all existing pretrained models!\nStart training from scratch...')
                logger.info('Deleting all existing pretrained models!\nStart training from scratch...')
                for ckpt in existing_ckpts:
                    os.remove(ckpt)
            return 0
        elif self.opt['train_strategy'].startswith('cont_train'):
            ED_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
                               len(re.findall(
                                   r'ED[.]epoch[-]\d+[.]config',
                                   fname)) > 0]
            if len(ED_config_paths) > 0:
                ED_config_path = ED_config_paths[0]
                self.opt['ED_eval_epoch'] = int(
                    re.findall(r'epoch[-](\d+)[.]config',
                               ED_config_path)[0])
                self.load_saved_model()
                print('Continue training at checkpoint: {}'.format(self.opt['ED_eval_epoch']))
                logger.info('Continue training at checkpoint: {}'.format(self.opt['ED_eval_epoch']))
                return self.opt['ED_eval_epoch'] + 1
            else:
                print(
                    'Saved model from hidden_eval.py is not found.\nStart training from scrach.'.format(
                        self.opt['train_strategy']))
                logger.info('Saved model from hidden_eval.py is not found.\nStart training from scrach.'.format(
                    self.opt['train_strategy']))
                return 0

    def train(self):
        print('Start training: ED model')
        logger.info('Start training: ED model')
        start_epoch = self.prepare_training()
        for epoch in range(start_epoch, start_epoch + self.opt['num_epoch']):
            self.train_epoch(epoch)

        out = '**************************************************\nEnd of training: ED model\nBest epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f})'.format(
            self.best_result['epoch'],
            self.best_result['en_p'],
            self.best_result['en_r'],
            self.best_result['en_f1']
        )
        if not self.opt['save_last_epoch']:
            print(out)
            logger.info(out)
            return self.best_result['epoch'], self.best_result['en_f1']
        else:
            print('Saving model...')
            logger.info('Saving model...')
            self.save_model(epoch)
            return -1, -1

    def record_scores(self, preds, labels, lang_weights):
        def compute_TP_FP_FN(ps, ls):
            TP_FN = torch.sum(ls.gt(0).long())
            TP_FP = torch.sum(ps.gt(0).long())
            mask = ls.gt(0).long()
            TP = torch.sum((ps == ls).long() * mask)

            return (TP, TP_FP, TP_FN)

        with torch.no_grad():
            lang_ws = lang_weights.data.cpu().numpy().tolist()
            en_indices = [k for k in range(len(lang_ws)) if lang_ws[k] == 1.]
            ar_indices = [k for k in range(len(lang_ws)) if lang_ws[k] < 1.]

            en_preds, en_labels = preds[en_indices, :], labels[en_indices, :]
            ar_preds, ar_labels = preds[ar_indices, :], labels[ar_indices, :]
            return {
                'english': compute_TP_FP_FN(en_preds, en_labels),
                'arabic': compute_TP_FP_FN(ar_preds, ar_labels)
            }

    def compute_scores(self, TP, TP_FP, TP_FN):
        TP = float(TP)
        TP_FP = float(TP_FP)
        TP_FN = float(TP_FN)

        prec = TP / TP_FP if TP_FP > 0 else 0.
        rec = TP / TP_FN if TP_FN > 0 else 0.
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return prec * 100., rec * 100., f1 * 100.

    def eval(self, epoch):
        self.model.eval()
        en_TP, en_TP_FP, en_TP_FN = 0, 0, 0

        for b_id, batch in enumerate(self.dev_iterator):
            batch = move_to_device(batch, self.opt['device'])
            _, probs, preds = self.model(batch)

            score = self.record_scores(preds, batch[-2], lang_weights=batch[-3])
            en_TP += score['english'][0]
            en_TP_FP += score['english'][1]
            en_TP_FN += score['english'][2]
            ###################################
        en_p, en_r, en_f1 = self.compute_scores(en_TP, en_TP_FP, en_TP_FN)
        out = '=' * 50 + '\n' + 'dev: p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
            en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
        )
        print(out)
        logger.info(out)
        if not self.opt['save_last_epoch'] and en_f1 > self.best_result['en_f1']:
            self.best_result['epoch'] = epoch
            self.best_result['en_p'] = en_p
            self.best_result['en_r'] = en_r
            self.best_result['en_f1'] = en_f1
            print('-> New best epoch')
            logger.info('-> New best epoch')
            if self.opt['delete_nonbest_ckpts']:
                print('Saving model...')
                logger.info('Saving model...')
                self.save_model(epoch)

        self.model.train()

    def eval_with_saved_model(self):
        saved_epoch = self.load_saved_model()
        self.model.eval()

        en_TP, en_TP_FP, en_TP_FN = 0, 0, 0

        for b_id, batch in enumerate(self.dev_iterator):
            batch = move_to_device(batch, self.opt['device'])
            _, probs, preds = self.model(batch)

            score = self.record_scores(preds, batch[-2], lang_weights=batch[-3])
            en_TP += score['english'][0]
            en_TP_FP += score['english'][1]
            en_TP_FN += score['english'][2]

            ###################################
        en_p, en_r, en_f1 = self.compute_scores(en_TP, en_TP_FP, en_TP_FN)

        print('ED model : Epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
            saved_epoch, en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
        ))

        logger.info(
            'ED model : Epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
                saved_epoch, en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
            )
        )

        return -1, en_f1

    def load_saved_model(self):
        ensure_dir('checkpoints')
        model_file = os.path.join('checkpoints',
                                  'ED-model.seed-{}.epoch-{}.saved'.format(
                                      self.opt['seed'],
                                      self.opt['ED_eval_epoch']
                                  ))

        epoch = -1
        print('Loading checkpoint: {}'.format(model_file))
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(model_file)
                # fairseq xlmr weirdly change the parameter names -> handle this as below:
                if "xlmr_embedding.model.encoder.sentence_encoder.embed_tokens.weight" in set(
                        self.model.state_dict().keys()):
                    converted_state_dict = {}
                    for key in checkpoint['model']:
                        converted_key = key.replace('xlmr_embedding.model.decoder.', 'xlmr_embedding.model.encoder.')
                        converted_state_dict[converted_key] = checkpoint['model'][key]
                    checkpoint['model'] = converted_state_dict

                self.model.load_state_dict(checkpoint['model'])
                if 'epoch' in checkpoint:
                    epoch = checkpoint['epoch']
                if 'opt' in checkpoint:
                    self.opt['lambda_mix'] = checkpoint['opt']['lambda_mix']
                print('Loaded!')
            except BaseException:
                print("Cannot load model from {}".format(model_file))
                exit(1)
        else:
            print('No checkpoints found!')
            exit(1)

        return epoch

    def save_model(self, epoch):
        ensure_dir('checkpoints')

        if self.opt['delete_nonbest_ckpts']:
            existing_ckpts = [fname for fname in get_files_in_dir('checkpoints') if
                              os.path.basename(fname).startswith('ED')]
            for ckpt in existing_ckpts:
                os.remove(ckpt)

        opt = {}
        for arg in self.opt:
            if arg not in ['biw2v_vecs', 'device']:
                opt[arg] = self.opt[arg]

        params = {
            'model': self.model.state_dict(),
            'opt': opt,
            'epoch': epoch
        }

        try:
            ckpt_fpath = os.path.join('checkpoints',
                                      'ED-model.seed-{}.epoch-{}.saved'.format(
                                          self.opt['seed'],
                                          epoch
                                      )
                                      )
            torch.save(params,
                       ckpt_fpath)
            print('... to: {}'.format(ckpt_fpath))
            logger.info('... to: {}'.format(ckpt_fpath))
            write_json(opt, write_path=os.path.join('checkpoints',
                                                    'ED.epoch-{}.config'.format(epoch)))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def predict(self, combined_task_inputs):
        with torch.no_grad():
            ED_preds, ED_probs, _ = self.model.predict(combined_task_inputs)
            return ED_preds, ED_probs
