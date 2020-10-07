# -*- coding: utf-8 -*-
# from python.clever.event_models.uoregon.models.pipeline._01.local_constants import *
#from fairseq.models.roberta import XLMRModel
from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.layers import DynamicLSTM, GCN, SelfAttention
#from nlplingo.oregon.event_models.uoregon.models.pipeline._01.iterators import upos_map, ner_map
from nlplingo.oregon.nlplingo.tasks.sequence.generator import upos_map, ner_map

from transformers import AutoConfig, XLMRobertaModel, XLMRobertaForMaskedLM, XLMRobertaForTokenClassification


class EDModelHF(nn.Module):
    def __init__(self, opt, label_map):
        print('========== ED_model_hf.EDModel.__init__ START ============')
        """ decode.bash
        upos_dim= 30
        self.rep_dim= 30
        use_ner= 0
        ner_dim= 30
        self.xlmr_dim= 768
        xlmr_model_dir= models/xlmr.base
        dropout_xlmr= 0.1
        num_last_layer_xlmr= 1
        hidden_dim= 200
        """
        super(EDModelHF, self).__init__()
        self.opt = opt
        self.label_map = label_map

        print('upos_dim=', self.opt['upos_dim'])

        self.upos_embedding = nn.Embedding(
            num_embeddings=len(upos_map),
            # TODO our upos_map in generator.py is the same len as theirs in iterators.py, so this is fine
            embedding_dim=self.opt['upos_dim'],
            padding_idx=0
        )
        self.rep_dim = self.opt['upos_dim']  # 30
        print('self.rep_dim=', self.rep_dim)

        print('use_ner=', self.opt['use_ner'])
        print('ner_dim=', self.opt['ner_dim'])
        if self.opt['use_ner']:
            self.ner_embedding = nn.Embedding(
                num_embeddings=len(ner_map),
                embedding_dim=self.opt['ner_dim'],
                padding_idx=0
            )
            self.rep_dim += self.opt['ner_dim']
        # *********************************************
        if 'base' in self.opt['xlmr_version']:
            self.xlmr_dim = 768
        elif 'large' in self.opt['xlmr_version']:
            self.xlmr_dim = 1024

        # self.xlmr_embedding = XLMRModel.from_pretrained(
        #     # os.path.join(WORKING_DIR, 'tools', 'xlmr_resources', self.opt['xlmr_version']),	# <==
        #     self.opt['xlmr_model_dir'],  # ==>
        #     checkpoint_file='model.pt')

        self.config = AutoConfig.from_pretrained(
            'xlm-roberta-base',
            num_labels=len(self.label_map),
            id2label = {str(v): k for k, v in self.label_map.items()},
            label2id = {k: v for k, v in self.label_map.items()},
            cache_dir=self.opt['cache_dir'],
            output_hidden_states=True
        )
        #self.xlmr_embedding = XLMRobertaModel(self.config)
        #self.xlmr_embedding = XLMRobertaForMaskedLM(self.config)
        self.xlmr_embedding = XLMRobertaForTokenClassification(self.config)

        print('self.xlmr_dim=', self.xlmr_dim)
        print('xlmr_model_dir=', self.opt['xlmr_model_dir'])
        print('dropout_xlmr=', self.opt['dropout_xlmr'])
        self.dropout = nn.Dropout(self.opt['dropout_xlmr'])  # 0.5

        print('num_last_layer_xlmr=', self.opt['num_last_layer_xlmr'])
        self.rep_dim += self.xlmr_dim * self.opt['num_last_layer_xlmr']  # 30 + 768 * 1
        # ********************************************
        self.self_att = SelfAttention(self.rep_dim, opt)

        self.gcn_layer = GCN(
            in_dim=self.rep_dim,
            hidden_dim=self.rep_dim,
            num_layers=2,
            opt=opt
        )

        print('biw2v_size=', opt['biw2v_size'])
        self.biw2v_embedding = nn.Embedding(
            opt['biw2v_size'],
            embedding_dim=300,
            padding_idx=PAD_ID
        )
        self.load_pretrained_biw2v()

        print('hidden_dim=', self.opt['hidden_dim'])
        self.fc_ED = nn.Sequential(
            nn.Linear(self.rep_dim * 2 + 300, self.opt['hidden_dim']),
            nn.ReLU(),
            # nn.Linear(self.opt['hidden_dim'], len(EVENT_MAP))	# <==
            nn.Linear(self.opt['hidden_dim'], len(label_map))  # ==>	TODO
        )
        print('========== ED_model.EDModel.__init__ END ============')

    def load_pretrained_biw2v(self):
        embed = self.biw2v_embedding
        vecs = self.opt['biw2v_vecs']
        pretrained = torch.from_numpy(vecs)
        embed.weight.data.copy_(pretrained)

    def get_xlmr_reps(self, inputs):
        print('============ ED_model_hf.get_xlmr_reps START =============')
        """
        xlmr_ids.shape= torch.Size([10, 53])
        retrieve_ids.shape= torch.Size([10, 33])
        type(all_hiddens)= <class 'list'>
        len(all_hiddens= 13
        all_hiddens[0].shape= torch.Size([10, 53, 768])
        all_hiddens[1].shape= torch.Size([10, 53, 768])
        all_hiddens[-1].shape= torch.Size([10, 53, 768])
        batch_size= 10
        len(all_hiddens)= 12
        self.opt['num_last_layer_xlmr']= 1
        used_layers= [11]
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        retrieve_reps.shape= torch.Size([33, 768])
        token_reps.shape= torch.Size([10, 33, 768])

        all_hiddens= [tensor([[[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
                  -1.6990e-01,  3.3114e-02],
                 [ 2.8519e-01,  2.1820e-01,  3.3214e-01,  ...,  3.9062e-01,
                   1.3669e-01,  1.4192e-01],
                 [ 6.8526e-02,  1.5400e-01,  1.7242e-02,  ..., -1.1426e-01,
                  -4.5462e-02,  5.1807e-02],
                 ...,
                 [ 3.1810e-01,  4.0966e-02,  2.1512e-01,  ...,  3.5518e-01,
                   2.6255e-01,  4.1006e-02],
                 [ 1.0284e-01,  5.7793e-02,  4.4513e-02,  ..., -2.3617e-01,
                   2.5314e-02,  6.0451e-02],
                 [ 6.8176e-03,  1.2782e-01,  7.2239e-02,  ..., -1.4924e-01,
                  -1.9298e-02,  1.6031e-01]],

                [[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
                  -1.6990e-01,  3.3114e-02],
                 [ 2.8519e-01,  2.1820e-01,  3.3214e-01,  ...,  3.9062e-01,
                   1.3669e-01,  1.4192e-01],
                 [ 6.8526e-02,  1.5400e-01,  1.7242e-02,  ..., -1.1426e-01,
                  -4.5462e-02,  5.1807e-02],
                 ...,
                 [ 1.3959e-01,  9.0699e-04,  2.0260e-01,  ...,  2.0667e-02,
                   3.6359e-01, -1.2589e-01],
                 [ 1.3913e-01,  6.6280e-02,  2.8022e-01,  ..., -2.7151e-02,
                   3.6584e-01, -6.2766e-02],
                 [ 1.2602e-01,  1.2431e-01,  2.7972e-01,  ..., -4.9168e-02,
                   4.1285e-01, -2.7115e-04]],

                [[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
                  -1.6990e-01,  3.3114e-02],
                 [-2.1162e-01, -1.2736e-02, -8.2769e-02,  ...,  1.2881e-01,
                   1.2014e-01,  2.7267e-01],
                 [-4.0390e-01, -6.7837e-02,  1.2579e-03,  ..., -6.0733e-03,
                   3.5541e-01, -1.9815e-01],
                 ...,
                 [ 1.3959e-01,  9.0699e-04,  2.0260e-01,  ...,  2.0667e-02,
                   3.6359e-01, -1.2589e-01],
                 [ 1.3913e-01,  6.6280e-02,  2.8022e-01,  ..., -2.7151e-02,
                   3.6584e-01, -6.2766e-02],
                 [ 1.2602e-01,  1.2431e-01,  2.7972e-01,  ..., -4.9168e-02,
                   4.1285e-01, -2.7115e-04]],

                 ...,
                 [ 4.9059e-01,  3.9329e-01, -1.3623e-01,  ..., -2.5431e-01,
                   1.1468e-01,  8.7181e-02],
                 [ 5.0399e-01,  3.8765e-01, -1.2510e-01,  ..., -3.0067e-01,
                   1.0453e-01,  1.6625e-01],
                 [ 5.4651e-01,  4.0442e-01, -1.6091e-01,  ..., -3.3413e-01,
                   5.9839e-02,  2.1487e-01]]], device='cuda:0')] 
        """
        xlmr_ids = inputs[0]
        input_mask = inputs[1]
        label_ids = inputs[2]
        retrieve_ids = inputs[4]
        print('xlmr_ids.shape=', xlmr_ids.shape)
        print('input_mask.shape=', input_mask.shape)
        print('label_ids.shape=', label_ids.shape)
        print('retrieve_ids.shape=', retrieve_ids.shape)

        print('xlmr_ids=', xlmr_ids)
        #print('attention_mask=', attention_mask)
        print('retrieve_ids=', retrieve_ids)

        # all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
        #inputs = {"input_ids": xlmr_ids, "attention_mask": input_mask, "labels": label_ids}
        #inputs = {"input_ids": xlmr_ids, "attention_mask": input_mask, "token_type_ids": (None)}
        inputs = {"input_ids": xlmr_ids}
        #inputs["token_type_ids"] = (None)  # XLM and RoBERTa don"t use segment_ids
        all_hiddens = self.xlmr_embedding(**inputs)

        #all_hiddens = self.xlmr_embedding.extract_features(xlmr_ids, return_all_hiddens=True)
        print('type(all_hiddens)=', type(all_hiddens))
        print('len(all_hiddens)=', len(all_hiddens))
        print('all_hiddens[0].shape=', all_hiddens[0].shape)
        print('len(all_hiddens[1])=', len(all_hiddens[1]))
        #print('all_hiddens[1].shape=', all_hiddens[1].shape)
        #print('all_hiddens[-1].shape=', all_hiddens[-1].shape)
        all_hiddens = all_hiddens[1]
        print('== all_hiddens = all_hiddens[1] ==')
        print('type(all_hiddens)=', type(all_hiddens))
        print('len(all_hiddens)=', len(all_hiddens))
        print('all_hiddens[0].shape=', all_hiddens[0].shape)
        print('all_hiddens[1].shape=', all_hiddens[1].shape)
        print('all_hiddens[-1].shape=', all_hiddens[-1].shape)

        all_hiddens = list(all_hiddens[1:])  # remove embedding layer

        token_reps = []

        batch_size, _ = xlmr_ids.shape
        print('batch_size=', batch_size)
        used_layers = list(range(len(all_hiddens)))[-self.opt['num_last_layer_xlmr']:]
        print('len(all_hiddens)=', len(all_hiddens))
        print("self.opt['num_last_layer_xlmr']=", self.opt['num_last_layer_xlmr'])
        print('used_layers=', used_layers)
        for example_id in range(batch_size):
            retrieved_reps = torch.cat([all_hiddens[layer_id][example_id][retrieve_ids[example_id]]
                                        for layer_id in used_layers], dim=1)  # [seq len, xlmr_dim x num last layers]
            print('retrieved_reps=', retrieved_reps)
            print('retrieve_reps.shape=', retrieved_reps.shape)
            token_reps.append(retrieved_reps)

        token_reps = torch.stack(token_reps, dim=0)  # [batch size, original seq len, xlmr_dim x num_layers]
        print('token_reps.shape=', token_reps.shape)
        print('============ ED_model.get_xlmr_reps END =============')
        return token_reps

    # def get_xlmr_reps(self, inputs):
    #     print('============ ED_model.get_xlmr_reps START =============')
    #     """
    #     xlmr_ids.shape= torch.Size([10, 53])
    #     retrieve_ids.shape= torch.Size([10, 33])
    #     type(all_hiddens)= <class 'list'>
    #     len(all_hiddens= 13
    #     all_hiddens[0].shape= torch.Size([10, 53, 768])
    #     all_hiddens[1].shape= torch.Size([10, 53, 768])
    #     all_hiddens[-1].shape= torch.Size([10, 53, 768])
    #     batch_size= 10
    #     len(all_hiddens)= 12
    #     self.opt['num_last_layer_xlmr']= 1
    #     used_layers= [11]
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     retrieve_reps.shape= torch.Size([33, 768])
    #     token_reps.shape= torch.Size([10, 33, 768])
    #
    #     all_hiddens= [tensor([[[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
    #               -1.6990e-01,  3.3114e-02],
    #              [ 2.8519e-01,  2.1820e-01,  3.3214e-01,  ...,  3.9062e-01,
    #                1.3669e-01,  1.4192e-01],
    #              [ 6.8526e-02,  1.5400e-01,  1.7242e-02,  ..., -1.1426e-01,
    #               -4.5462e-02,  5.1807e-02],
    #              ...,
    #              [ 3.1810e-01,  4.0966e-02,  2.1512e-01,  ...,  3.5518e-01,
    #                2.6255e-01,  4.1006e-02],
    #              [ 1.0284e-01,  5.7793e-02,  4.4513e-02,  ..., -2.3617e-01,
    #                2.5314e-02,  6.0451e-02],
    #              [ 6.8176e-03,  1.2782e-01,  7.2239e-02,  ..., -1.4924e-01,
    #               -1.9298e-02,  1.6031e-01]],
    #
    #             [[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
    #               -1.6990e-01,  3.3114e-02],
    #              [ 2.8519e-01,  2.1820e-01,  3.3214e-01,  ...,  3.9062e-01,
    #                1.3669e-01,  1.4192e-01],
    #              [ 6.8526e-02,  1.5400e-01,  1.7242e-02,  ..., -1.1426e-01,
    #               -4.5462e-02,  5.1807e-02],
    #              ...,
    #              [ 1.3959e-01,  9.0699e-04,  2.0260e-01,  ...,  2.0667e-02,
    #                3.6359e-01, -1.2589e-01],
    #              [ 1.3913e-01,  6.6280e-02,  2.8022e-01,  ..., -2.7151e-02,
    #                3.6584e-01, -6.2766e-02],
    #              [ 1.2602e-01,  1.2431e-01,  2.7972e-01,  ..., -4.9168e-02,
    #                4.1285e-01, -2.7115e-04]],
    #
    #             [[-1.5241e-01,  1.5346e-01, -1.4166e-01,  ...,  5.2533e-02,
    #               -1.6990e-01,  3.3114e-02],
    #              [-2.1162e-01, -1.2736e-02, -8.2769e-02,  ...,  1.2881e-01,
    #                1.2014e-01,  2.7267e-01],
    #              [-4.0390e-01, -6.7837e-02,  1.2579e-03,  ..., -6.0733e-03,
    #                3.5541e-01, -1.9815e-01],
    #              ...,
    #              [ 1.3959e-01,  9.0699e-04,  2.0260e-01,  ...,  2.0667e-02,
    #                3.6359e-01, -1.2589e-01],
    #              [ 1.3913e-01,  6.6280e-02,  2.8022e-01,  ..., -2.7151e-02,
    #                3.6584e-01, -6.2766e-02],
    #              [ 1.2602e-01,  1.2431e-01,  2.7972e-01,  ..., -4.9168e-02,
    #                4.1285e-01, -2.7115e-04]],
    #
    #              ...,
    #              [ 4.9059e-01,  3.9329e-01, -1.3623e-01,  ..., -2.5431e-01,
    #                1.1468e-01,  8.7181e-02],
    #              [ 5.0399e-01,  3.8765e-01, -1.2510e-01,  ..., -3.0067e-01,
    #                1.0453e-01,  1.6625e-01],
    #              [ 5.4651e-01,  4.0442e-01, -1.6091e-01,  ..., -3.3413e-01,
    #                5.9839e-02,  2.1487e-01]]], device='cuda:0')]
    #     """
    #     xlmr_ids = inputs[0]
    #     retrieve_ids = inputs[2]
    #     print('xlmr_ids.shape=', xlmr_ids.shape)
    #     print('retrieve_ids.shape=', retrieve_ids.shape)
    #
    #     # all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
    #     all_hiddens = self.xlmr_embedding.extract_features(xlmr_ids, return_all_hiddens=True)
    #     print('type(all_hiddens)=', type(all_hiddens))
    #     print('len(all_hiddens=', len(all_hiddens))
    #     print('all_hiddens[0].shape=', all_hiddens[0].shape)
    #     print('all_hiddens[1].shape=', all_hiddens[1].shape)
    #     print('all_hiddens[-1].shape=', all_hiddens[-1].shape)
    #
    #     all_hiddens = list(all_hiddens[1:])  # remove embedding layer
    #
    #     token_reps = []
    #
    #     batch_size, _ = xlmr_ids.shape
    #     print('batch_size=', batch_size)
    #     used_layers = list(range(len(all_hiddens)))[-self.opt['num_last_layer_xlmr']:]
    #     print('len(all_hiddens)=', len(all_hiddens))
    #     print("self.opt['num_last_layer_xlmr']=", self.opt['num_last_layer_xlmr'])
    #     print('used_layers=', used_layers)
    #     for example_id in range(batch_size):
    #         retrieved_reps = torch.cat([all_hiddens[layer_id][example_id][retrieve_ids[example_id]]
    #                                     for layer_id in used_layers], dim=1)  # [seq len, xlmr_dim x num last layers]
    #         print('retrieve_reps.shape=', retrieved_reps.shape)
    #         token_reps.append(retrieved_reps)
    #
    #     token_reps = torch.stack(token_reps, dim=0)  # [batch size, original seq len, xlmr_dim x num_layers]
    #     print('token_reps.shape=', token_reps.shape)
    #     print('============ ED_model.get_xlmr_reps END =============')
    #     return token_reps

    def forward(self, inputs):
        print('=============== ED_model_hf.forward START ============')
        xlmr_ids, input_mask, label_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights, ED_labels, pad_masks = inputs
        print('xlmr_ids.shape=', xlmr_ids.shape)
        print('input_mask.shape=', input_mask.shape)
        print('label_ids.shape=', label_ids.shape)
        print('biw2v_ids.shape=', biw2v_ids.shape)
        print('retrieve_ids.shape=', retrieve_ids.shape)
        print('upos_ids.shape=', upos_ids.shape)
        print('xpos_ids.shape=', xpos_ids.shape)
        print('head_ids.shape=', head_ids.shape)
        print('deprel_ids.shape=', deprel_ids.shape)
        print('ner_ids.shape=', ner_ids.shape)
        print('lang_weights.shape=', lang_weights.shape)
        print('ED_labels.shape=', ED_labels.shape)
        print('pad_masks.shape=', pad_masks.shape)

        """
        xlmr_ids.shape= torch.Size([16, 63])
        biw2v_ids.shape= torch.Size([16, 51])
        retrieve_ids.shape= torch.Size([16, 51])
        upos_ids.shape= torch.Size([16, 51])
        xpos_ids.shape= torch.Size([16, 51])
        head_ids.shape= torch.Size([16, 51])
        deprel_ids.shape= torch.Size([16, 51])
        ner_ids.shape= torch.Size([16, 51])
        lang_weights.shape= torch.Size([16])
        ED_labels.shape= torch.Size([16, 51])
        pad_masks.shape= torch.Size([16, 51])
        token_masks.shape= torch.Size([16, 51])
        upos_reps.shape= torch.Size([16, 51, 30])
        """

        token_masks = pad_masks.eq(0).float()
        print('token_masks.shape=', token_masks.shape)
        # ****** word embeddings ********
        upos_reps = self.upos_embedding(upos_ids)  # [batch size, seq len, upos dim]
        print('upos_reps.shape=', upos_reps.shape)

        word_feats = []
        word_feats.append(upos_reps)

        if self.opt['use_ner']:
            ner_reps = self.ner_embedding(ner_ids)
            word_feats.append(ner_reps)

        word_embeds = self.get_xlmr_reps(inputs)  # [batch size, seq len, xlmr dim]
        """ from above self.get_xlmr_reps()
        xlmr_ids.shape= torch.Size([16, 63])
        retrieve_ids.shape= torch.Size([16, 51])
        type(all_hiddens)= <class 'list'>
        len(all_hiddens= 13
        all_hiddens[0].shape= torch.Size([16, 63, 768])
        all_hiddens[1].shape= torch.Size([16, 63, 768])
        all_hiddens[-1].shape= torch.Size([16, 63, 768])
        batch_size= 16
        len(all_hiddens)= 12
        self.opt['num_last_layer_xlmr']= 1
        used_layers= [11]
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        retrieve_reps.shape= torch.Size([51, 768])
        token_reps.shape= torch.Size([16, 51, 768])
        """

        """
        word_embeds.shape= torch.Size([16, 51, 768])
        word_embeds.shape= torch.Size([16, 51, 768])
        word_reps.shape= torch.Size([16, 51, 798])
        """
        print('word_embeds.shape=', word_embeds.shape)
        word_embeds = self.dropout(word_embeds)
        print('word_embeds.shape=', word_embeds.shape)

        word_feats.append(word_embeds)

        word_reps = torch.cat(word_feats, dim=2)
        print('word_reps.shape=', word_reps.shape)
        # *******************************
        """
        In below self.self_att()
        input_masks.shape= torch.Size([16, 51])
        slf_attn_mask.shape= torch.Size([16, 51, 51])
        non_pad_mask.shape= torch.Size([16, 51, 1])
        enc_output.shape= torch.Size([16, 51, 798])
        position_embed_for_satt= 1
        position_ids.shape= torch.Size([16, 51])
        enc_output.shape= torch.Size([16, 51, 798])
        """
        satt_reps, att_weights = self.self_att(word_reps, pad_masks)

        """
        satt_reps.shape= torch.Size([16, 51, 798])
        att_weights.shape= torch.Size([16, 51, 51])
        adj.shape= torch.Size([16, 51, 51])
        gcn_reps.shape= torch.Size([16, 51, 798])
        muse_reps.shape= torch.Size([16, 51, 300])
        final_reps.shape= torch.Size([16, 51, 1896])
        logits.shape= torch.Size([16, 51, 16])
        loss= tensor(2.8248, device='cuda:0', grad_fn=<DivBackward0>)
        probs.shape= torch.Size([16, 51, 16])
        preds.shape= torch.Size([16, 51])
        """
        print('satt_reps.shape=', satt_reps.shape)
        print('att_weights.shape=', att_weights.shape)

        adj = get_full_adj(head_ids, pad_masks, self.opt['device'])
        print('adj.shape=', adj.shape)
        gcn_reps, _ = self.gcn_layer(word_reps, adj)
        print('gcn_reps.shape=', gcn_reps.shape)

        muse_reps = self.biw2v_embedding(biw2v_ids)
        print('muse_reps.shape=', muse_reps.shape)
        final_reps = torch.cat(
            [satt_reps, gcn_reps, muse_reps],
            dim=2
        )
        print('final_reps.shape=', final_reps.shape)

        logits = self.fc_ED(final_reps)  # [batch size, seq len, 16]
        print('logits.shape=', logits.shape)
        loss, probs, preds = compute_batch_loss(logits, ED_labels, token_masks, instance_weights=lang_weights)
        print('loss=', loss)
        print('probs.shape=', probs.shape)
        print('preds.shape=', preds.shape)
        print('=============== ED_model_hf.forward END ============')
        return loss, probs, preds

    def predict(self, combined_task_inputs):
        xlmr_ids, input_mask, label_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, eid, pad_masks = combined_task_inputs
        token_masks = pad_masks.eq(0).float()  # 1.0 if true token, else 0
        print('========== ED_model.predict START ===============')
        """
        token_masks.shape= torch.Size([10, 33])
        upos_reps.shape= torch.Size([10, 33, 30])
        """
        print('token_masks.shape=', token_masks.shape)
        """
        xlmr_ids.shape= torch.Size([10, 53])
        biw2v_ids.shape= torch.Size([10, 33])
        retrieve_ids.shape= torch.Size([10, 33])
        upos_ids.shape= torch.Size([10, 33])
        xpos_ids.shape= torch.Size([10, 33])
        head_ids.shape= torch.Size([10, 33])
        deprel_ids.shape= torch.Size([10, 33])
        ner_ids.shape= torch.Size([10, 33])
        eid.shape= torch.Size([10])
        pad_masks.shape= torch.Size([10, 33])

        xlmr_ids= tensor([[     0,      6,      5,  90621,  47229,    250,    181,   5273,  10408,
                   6267,   4039,  31245,  71633,   2620,  18684,   6466,   7233,    250,
                    240, 102468,    368,      6, 185701,  35618,  18004, 159565,  97288,
                  41468,    152,     94,  13231,   3108,    746,  14272,   3070, 102935,
                   2103, 153872,    767, 186386,  12581,  30039,    230,  59721, 148726,
                    755,    230,   6816,   1692,    340,      6,      5,      2],
                [     0,      6,      5,  45869,  53929,  10286, 112847,    593,  50221,
                 139152,  46416,    179,  83001,  95451, 104042,    240,  13875,  13874,
                  18004,  39865,   3363,  93319, 136295, 109177,    240,  81881, 189757,
                  81972,  43060,    230,  11115,  33018,    702,  48102,  46408,  73279,
                     94,   9580, 199317,  73942, 160700,  35508,    340,      6,      5,
                      2,      0,      0,      0,      0,      0,      0,      0],
                [     0,   4003,  20621,    862,  18173,  30099,   7624,    906, 141538,
                    755,    556,  48964,  61501,     65, 123290, 164456,    230,   4569,
                  74602,    240, 169348,  47769,  48387,  47769,  16994,    396, 113409,
                 216336,    755,      6,  92127,  36435,  52316,  23628,     65,  32634,
                   1195, 110813,    240,  34708, 201174,      6,      5,      2,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,   6625,    665,  87151,  73397,    906, 129382,  10731,  87509,
                      6, 114378,  13620,   3015,  96629,  92564,   5202,   3015,  96629,
                  92564,   3108,  59545,    665, 101375,    258,  25198,  13231,   4003,
                   3518, 123506,    906,  24832,    755, 194558,    250,  19636,   3518,
                  98058,   3202,   1692,      6,      5,      2,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,      6,      5,  37705, 112600,    376,  40743,   5202,  43228,
                  12323,  48483,   9787,   1325,   1855,   5081,   2044,    826,      6,
                 110351,    176,    230,      6, 163970,  19089,  47600,  96517,  16452,
                    412,   6963,   1533,    862,  18740,  13029,  66087,   1365,      6,
                 116337,   1692,      6,      5,      2,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,      6, 104815,  53411,      6, 130825,      6,  22650,  54563,
                    240,      6,  97927,  10691,    240,  65525,      6, 224157,    665,
                  77358,    250,  27952,  35180, 160769,  22366,  19931, 101632,    648,
                  15776,    179,  26430,  70153,  12337,   2977,    240, 103919,      6,
                      5,      2,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,   1333, 214363,  96517,  22327,   8039,   3088,   1335,  51218,
                    902, 177421, 154597,   1533, 146142,    755,    230, 206210,  15330,
                  69294,    240,    359, 169368,   4040,  14924,   8428,  35862,  10691,
                  15493,  72317,    179,  12888,      6,      5,      2,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,   6625, 106969,  24094,    917,  10913,  10937,      6,  83188,
                  13759,    240,  93584,   1335,  86401,  24537,   5706,   5202,  24094,
                 208045,    862, 155500,  48707,   8665,  45089, 121818,  84341,    412,
                 220818,      6,      5,      2,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0, 105285,    368,  35000,    230,  12584,    230,   4382,  29928,
                    240, 141677,    250,  18740,  54610,  60930,    240,  30506,      6,
                  48699, 140252,    258,    556,      6, 164072,  12589,  96517,      6,
                      5,      2,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0],
                [     0,   6625,  55468,    900,   1705, 124630, 151721,   5202, 234180,
                   3518,  48633,     94,  73441,  23579,    376,  18486, 122608,    340,
                    240,  37160,  11945,    240,  72647, 120465,   5784, 133131,      6,
                      5,      2,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0,      0,      0]],
               device='cuda:0')
        biw2v_ids= tensor([[     6, 114225, 113937, 128409,      1,   1443, 113675, 113666, 163097,
                 117713,      1, 126519,      1, 113266, 114068,      1,     50, 176957,
                      1, 113209, 127252, 113173, 113584, 120372, 126250, 113253, 113470,
                 113165, 117399, 113165, 119105, 177487,      6],
                [     6, 113782,      1, 123638,      1,      1, 131450, 113546,      1,
                 116631, 113266, 114666,      1, 125284,      1, 115773, 117903, 124178,
                 113165, 113254,      1,    395, 113309,      1, 176957, 113203,      1,
                      1,      1, 177487,      6,      0,      0],
                [113216, 113383,      1, 113448, 119129, 113264, 120182, 113167, 242005,
                 137590,      1, 113165, 137330,      1,      1, 234085, 179317, 115962,
                 162598, 114539, 114727, 114453,      1,      1, 114368, 114375,      6,
                      0,      0,      0,      0,      0,      0],
                [113306, 115538, 113453,      1, 113183,      1, 120945,      1,      1,
                      1, 113209, 114243,      1, 113395,      1, 113216, 169807, 117207,
                 116183,      1, 192377, 121340,      6,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [     6, 114821, 163342,      1, 130192,      1, 150092,      1, 191632,
                 113165,      1, 123445, 113381, 114054, 193520, 113200,      1, 113604,
                 134623, 113170, 122650,      6,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [     1, 113621,      1, 123566,      1, 114004,      1,      1, 117675,
                 122110, 113447, 113985, 118076, 190493, 171192,      1, 113939,      1,
                 118621,      6,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [     1, 113381, 121106,      1, 118400, 113950, 113725, 113200,      1,
                 113165, 115868, 168565,      1, 117866, 113179,      1, 151891,      6,
                      0,      0,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [113306, 116827, 113219, 154925, 148370,      1, 113596,      1,      1,
                      1, 113219, 115695,      1, 114991, 113407, 156650, 115304, 113180,
                      1,      6,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [125488, 113398, 113165,      1, 113165,   9086, 116472,      1, 119791,
                 113604, 114109, 115046,      1, 123924, 117632, 119425, 113167, 120572,
                 113381,      6,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0],
                [113306, 116408, 135535,      1,      1,      1, 137810, 176957, 129403,
                 123127, 177487,      1, 113249, 113236,      1, 113558,      1, 113199,
                 113472,      6,      0,      0,      0,      0,      0,      0,      0,
                      0,      0,      0,      0,      0,      0]], device='cuda:0')
        retrieve_ids= tensor([[ 2,  3,  4,  6,  8, 10, 11, 12, 13, 16, 18, 19, 22, 24, 25, 26, 28, 29,
                 30, 31, 32, 34, 35, 36, 38, 40, 41, 42, 43, 46, 47, 49, 51],
                [ 2,  3,  4,  6,  8, 10, 12, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27,
                 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 44,  0,  0],
                [ 1,  2,  3,  4,  5,  6,  7, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26,
                 27, 30, 32, 33, 34, 38, 39, 40, 42,  0,  0,  0,  0,  0,  0],
                [ 1,  2,  4,  5,  7,  8, 10, 12, 15, 16, 19, 20, 21, 24, 25, 26, 27, 29,
                 32, 34, 35, 37, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 2,  3,  4,  7,  8, 10, 11, 14, 18, 20, 22, 23, 25, 26, 27, 29, 30, 31,
                 32, 34, 36, 39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 2,  3,  5,  7,  9, 11, 13, 14, 16, 17, 20, 21, 22, 23, 27, 30, 32, 33,
                 34, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  3,  4,  7,  8, 10, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 28, 32,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  2,  3,  4,  8, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 24, 25,
                 26, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 23,
                 25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  2,  4,  6,  7,  8,  9, 11, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24,
                 25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
               device='cuda:0')
        upos_ids= tensor([[ 7,  8,  4, 18, 18,  6,  4, 18, 18,  9,  2, 18, 18,  2,  6,  4,  7,  7,
                 14, 13,  8,  3,  4, 18,  8,  4,  9,  2,  4,  2, 18,  7,  7],
                [ 7, 18,  8,  4, 18, 18, 18,  9,  2, 18,  2,  4,  4,  9,  2,  4,  4,  9,
                  2,  4,  4,  6,  4, 18,  7,  4,  9, 18,  9,  7,  7,  0,  0],
                [ 8,  8, 14,  8,  4,  4,  4,  2, 18, 18,  9,  2,  4,  2,  4, 18, 18, 18,
                  4, 18, 18, 18, 18,  2,  4,  4,  7,  0,  0,  0,  0,  0,  0],
                [ 8,  4,  2,  4,  9,  9, 18, 18, 14, 18, 13, 12,  9, 14, 14,  8,  4,  9,
                  6, 14, 18,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 7,  8, 18, 14,  4,  4, 18, 18, 18,  2,  4,  4,  4,  2, 18,  2, 14,  2,
                  4,  2, 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 8,  4,  4, 18,  2,  4,  2,  4,  4,  9, 18,  4,  4, 18,  9, 18, 18,  2,
                  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 8,  4,  4,  2,  4,  9,  9,  2,  4,  2,  4,  9,  2,  4,  2,  9, 18,  7,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 8,  4,  4, 18, 18,  2,  4,  2, 18, 14,  4,  8, 14,  4,  9, 18,  4,  4,
                 18,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 8,  4,  2,  9,  2,  6,  4,  2,  4,  2,  4, 18,  2,  4,  8,  4,  2,  4,
                  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 8,  4, 18,  9, 14, 18, 18,  7,  4,  4,  7,  2,  4,  4,  2,  4,  9,  2,
                  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
               device='cuda:0')
        xpos_ids= tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
        head_ids= tensor([[ 2,  0,  2,  3,  4,  2,  6,  9,  6,  2, 12, 10, 12, 15, 10, 15,  2, 21,
                 21, 21,  2, 23, 21, 25, 21, 25, 26, 29, 26, 31, 25, 25,  2],
                [ 3,  3,  0,  3,  6,  3,  6,  7, 10,  6, 12,  3, 12, 13, 16, 12, 16, 17,
                 20, 18, 20, 20, 22, 23, 26, 20, 26, 26, 28,  3,  3,  0,  0],
                [ 0,  1,  4,  2,  4,  5,  6,  9,  5,  9, 10, 13, 10, 15,  4, 17,  9, 17,
                 18, 21, 19, 21, 19, 25, 23, 25,  1,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  4,  2,  4,  4,  2,  7, 13, 13, 13, 13,  1, 16, 14, 13, 16, 17,
                 18, 21, 16, 21,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 2,  0,  2,  8,  8,  5,  5,  2,  8, 11,  8, 11, 12, 15, 11, 19, 19, 19,
                 11, 21, 19,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  2,  3,  6,  3,  8,  1,  8,  8, 12,  1, 12, 13, 13, 17, 15, 19,
                 12,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  1,  5,  1,  5,  5,  9,  1, 11,  9, 11, 14, 11, 16, 14, 16,  1,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  2,  5,  2,  7,  1,  9,  7, 12, 12,  1, 16, 16, 14, 12, 16, 17,
                 18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  4,  2,  6,  4,  6,  9,  1, 11,  9, 11, 14,  1, 14, 15, 18, 16,
                 18,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  1,  2,  2,  7,  7,  1,  7,  1,  9, 13, 13,  9, 13, 16, 13, 16, 19,
                  9,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
               device='cuda:0')
        deprel_ids= tensor([[ 7,  9,  8,  5, 24,  4,  5, 20,  5, 18,  2, 13, 24,  2,  4,  5,  7,  7,
                 17, 13, 26,  3,  8, 20, 12,  8, 15,  2,  5,  2,  4,  7,  7],
                [ 7, 16,  9,  8,  2,  4,  5, 15,  2,  5,  2,  4,  5, 15,  2,  4,  5, 15,
                  2,  4,  5,  6,  5, 20,  7,  5, 15,  5, 15,  7,  7,  0,  0],
                [ 9, 18, 17, 27,  8,  5,  5,  2,  5,  5, 15,  2,  5,  2,  4, 20, 12,  5,
                  5,  2,  5, 24, 20,  2,  4,  5,  7,  0,  0,  0,  0,  0,  0],
                [ 9,  8,  2,  5, 15, 15,  5, 24, 17,  8, 13, 25, 22, 17, 28, 26,  8, 15,
                  6, 20, 12, 10,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 7,  9,  8, 17,  8,  5,  5, 10, 10,  2,  4,  5,  5,  2, 13,  2, 17,  2,
                  5,  2,  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 9,  8,  5,  5,  2,  5,  2,  4,  5, 15,  2,  4,  5,  5, 15,  2,  5,  2,
                  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 9,  8, 10,  2,  4, 15, 15,  2,  4,  2,  5, 15,  2,  5,  2, 15, 20,  7,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 9,  8,  5,  5,  5,  2,  4,  2,  5, 17,  8, 22, 17,  8, 15, 10, 10,  5,
                  5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 9,  8,  2, 15,  2,  6,  5,  2,  4,  2,  5,  5,  2,  4, 14, 10,  2,  5,
                  5,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 9,  8,  5, 15, 17,  5, 10,  7, 10,  5,  7,  2,  5,  5,  2,  5, 15,  2,
                  4,  7,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
               device='cuda:0')
        ner_ids= tensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
        eid= tensor([1., 3., 6., 5., 2., 9., 7., 8., 4., 0.], device='cuda:0')
        pad_masks= tensor([[False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False, False],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False, False,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                 False, False,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False,  True,  True,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True],
                [False, False, False, False, False, False, False, False, False, False,
                 False, False, False, False, False, False, False, False, False, False,
                  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                  True,  True,  True]], device='cuda:0')
        token_masks= tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        """

        # ****** word embeddings ********
        upos_reps = self.upos_embedding(upos_ids)  # [batch size, seq len, upos dim]
        print('upos_reps.shape=', upos_reps.shape)

        word_feats = []
        word_feats.append(upos_reps)

        if self.opt['use_ner']:
            ner_reps = self.ner_embedding(ner_ids)
            word_feats.append(ner_reps)

        word_embeds = self.get_xlmr_reps(combined_task_inputs)  # [batch size, seq len, xlmr dim]
        """
        word_embeds.shape= torch.Size([10, 33, 768])
        word_embeds.shape= torch.Size([10, 33, 768])
        word_reps.shape= torch.Size([10, 33, 798])
        """
        print('word_embeds.shape=', word_embeds.shape)
        word_embeds = self.dropout(word_embeds)
        print('word_embeds.shape=', word_embeds.shape)

        word_feats.append(word_embeds)

        word_reps = torch.cat(word_feats, dim=2)  # should be: [batch size, seq len, upos_dim + xlmr_dim]
        print('word_reps.shape=', word_reps.shape)
        # *******************************

        """ When I call self.self_att() below
        input_masks.shape= torch.Size([10, 33])
        slf_attn_mask.shape= torch.Size([10, 33, 33])
        non_pad_mask.shape= torch.Size([10, 33, 1])
        enc_output.shape= torch.Size([10, 33, 798])
        position_embed_for_satt= 1
        position_ids.shape= torch.Size([10, 33])
        enc_output.shape= torch.Size([10, 33, 798])
        """
        satt_reps, att_weights = self.self_att(word_reps, pad_masks)

        """
        satt_reps.shape= torch.Size([10, 33, 798]) att_weights.shape= torch.Size([10, 33, 33])
        adj.shape= torch.Size([10, 33, 33])
        gcn_reps.shape= torch.Size([10, 33, 798])
        muse_reps.shape= torch.Size([10, 33, 300])
        final_reps.shape= torch.Size([10, 33, 1896])
        logits.shape= torch.Size([10, 33, 16])
        preds.shape= torch.Size([10, 33])
        probs.shape= torch.Size([10, 33, 16])
        """
        print('satt_reps.shape=', satt_reps.shape, 'att_weights.shape=', att_weights.shape)

        adj = get_full_adj(head_ids, pad_masks, self.opt['device'])
        print('adj.shape=', adj.shape)
        gcn_reps, _ = self.gcn_layer(word_reps, adj)
        print('gcn_reps.shape=', gcn_reps.shape)
        muse_reps = self.biw2v_embedding(biw2v_ids)
        print('muse_reps.shape=', muse_reps.shape)

        final_reps = torch.cat(
            [satt_reps, gcn_reps, muse_reps],
            dim=2
        )
        print('final_reps.shape=', final_reps.shape)

        logits = self.fc_ED(final_reps)  # [batch size, seq len, 16]
        print('logits.shape=', logits.shape)
        preds = torch.argmax(logits, dim=2).long() * token_masks.long()
        print('preds.shape=', preds.shape)

        probs = torch.softmax(logits, dim=2)  # [batch size, seq len, num classes]
        print('probs.shape=', probs.shape)
        """
        preds.shape= torch.Size([10, 33])
        probs.shpae= torch.Size([10, 33, 16])
        token_masks.shape= torch.Size([10, 33])

        preds= tensor([[ 0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0, 15,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0, 10, 14,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0, 13,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,  0,  0],
                [14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0, 14,  0,  0,  0,  0,  0,  0, 14,  0,  5,  0,  0,  0,  0,  0,  0, 14,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  5,  0,  0,  0,  5,  5,  0,  0, 14,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0, 13,  0,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [14,  0,  0,  0,  0,  0, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [13,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  7,  0,  5,  0,  0,
                  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
               device='cuda:0')
        probs= tensor([[[9.9974e-01, 1.8049e-07, 1.8267e-08,  ..., 2.0810e-05,
                  5.8121e-05, 1.7711e-05],
                 [4.0915e-03, 8.9600e-07, 2.5587e-07,  ..., 3.6581e-04,
                  9.9335e-01, 2.5578e-04],
                 [9.9981e-01, 6.6693e-08, 7.0893e-09,  ..., 2.9757e-05,
                  3.0001e-05, 1.2025e-05],
                 ...,
                 [9.9945e-01, 2.5450e-07, 3.3773e-08,  ..., 6.2535e-05,
                  3.2478e-05, 1.0961e-04],
                 [9.9969e-01, 1.9118e-07, 1.9771e-08,  ..., 1.9469e-05,
                  4.6727e-05, 3.4929e-05],
                 [9.9975e-01, 1.6357e-07, 1.6790e-08,  ..., 1.7160e-05,
                  4.9481e-05, 2.1778e-05]],

                [[9.9969e-01, 2.4833e-07, 2.8136e-08,  ..., 3.9068e-05,
                  6.1829e-05, 1.7187e-05],
                 [9.9979e-01, 1.0620e-07, 1.4160e-08,  ..., 3.8181e-05,
                  3.3937e-05, 8.8798e-06],
                 [1.9942e-01, 1.2716e-05, 7.3964e-06,  ..., 3.7772e-02,
                  2.7622e-02, 1.6138e-03],
                 ...,
                 [9.9968e-01, 2.4205e-07, 2.7755e-08,  ..., 3.4218e-05,
                  4.3995e-05, 2.6076e-05],
                 [1.1816e-01, 3.0641e-02, 2.3439e-02,  ..., 7.2712e-02,
                  8.3229e-02, 7.6786e-02],
                 [1.1816e-01, 3.0641e-02, 2.3439e-02,  ..., 7.2712e-02,
                  8.3229e-02, 7.6786e-02]],

                [[9.9974e-01, 4.8116e-08, 7.5416e-09,  ..., 5.9740e-05,
                  7.3432e-05, 7.0663e-06],
                 [9.9976e-01, 5.1815e-08, 8.4192e-09,  ..., 4.0064e-05,
                  4.6581e-05, 6.7058e-06],
                 [9.9986e-01, 3.8352e-08, 5.4001e-09,  ..., 2.8621e-05,
                  2.2455e-05, 4.1249e-06],
                 ...,
                 [1.1728e-01, 3.1099e-02, 2.3771e-02,  ..., 7.2479e-02,
                  8.2832e-02, 7.6790e-02],
                 [1.1728e-01, 3.1099e-02, 2.3771e-02,  ..., 7.2479e-02,
                  8.2832e-02, 7.6790e-02],
                 [1.1728e-01, 3.1099e-02, 2.3771e-02,  ..., 7.2479e-02,
                  8.2832e-02, 7.6790e-02]],

                ...,

                [[3.7560e-03, 1.7332e-06, 7.1111e-07,  ..., 6.5631e-04,
                  9.9067e-01, 4.6437e-04],
                 [9.9888e-01, 3.3755e-07, 6.8032e-08,  ..., 1.0844e-04,
                  2.3969e-04, 1.1803e-04],
                 [9.9945e-01, 1.6661e-07, 2.9271e-08,  ..., 5.2645e-05,
                  8.6269e-05, 7.7344e-05],
                 ...,
                 [1.2599e-01, 2.6760e-02, 2.0231e-02,  ..., 7.5109e-02,
                  8.6246e-02, 7.6045e-02],
                 [1.2599e-01, 2.6760e-02, 2.0231e-02,  ..., 7.5109e-02,
                  8.6246e-02, 7.6045e-02],
                 [1.2599e-01, 2.6760e-02, 2.0231e-02,  ..., 7.5109e-02,
                  8.6246e-02, 7.6045e-02]],

                [[1.2495e-01, 3.4042e-06, 1.7341e-06,  ..., 8.2814e-01,
                  2.5807e-02, 7.6088e-03],
                 [9.9954e-01, 8.0825e-08, 1.2043e-08,  ..., 1.4239e-04,
                  5.0908e-05, 9.5018e-06],
                 [9.9973e-01, 3.7680e-08, 4.6598e-09,  ..., 5.9885e-05,
                  3.4976e-05, 6.5925e-06],
                 ...,
                 [1.2006e-01, 2.9710e-02, 2.2615e-02,  ..., 7.3734e-02,
                  8.3705e-02, 7.6766e-02],
                 [1.2006e-01, 2.9710e-02, 2.2615e-02,  ..., 7.3734e-02,
                  8.3705e-02, 7.6766e-02],
                 [1.2006e-01, 2.9710e-02, 2.2615e-02,  ..., 7.3734e-02,
                  8.3705e-02, 7.6766e-02]],

                [[2.9218e-03, 1.2115e-06, 4.4520e-07,  ..., 7.4744e-04,
                  9.9328e-01, 2.7579e-04],
                 [9.9936e-01, 2.0276e-07, 2.9777e-08,  ..., 7.2475e-05,
                  2.7122e-04, 3.7922e-05],
                 [9.9962e-01, 1.7503e-07, 2.2917e-08,  ..., 4.1906e-05,
                  6.1219e-05, 2.5442e-05],
                 ...,
                 [1.1812e-01, 3.0612e-02, 2.3442e-02,  ..., 7.2634e-02,
                  8.3626e-02, 7.6709e-02],
                 [1.1812e-01, 3.0612e-02, 2.3442e-02,  ..., 7.2634e-02,
                  8.3626e-02, 7.6709e-02],
                 [1.1812e-01, 3.0612e-02, 2.3442e-02,  ..., 7.2634e-02,
                  8.3626e-02, 7.6709e-02]]], device='cuda:0')
        token_masks= tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
               device='cuda:0')
        """
        print('========== ED_model.predict END ===============')
        return preds, probs, token_masks
