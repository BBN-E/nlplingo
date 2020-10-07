# -*- coding: utf-8 -*-
#from nlplingo.common.utils import DEPREL_TO_ID	# ==>

from nlplingo.oregon.event_models.uoregon.models.pipeline._01.local_constants import *
from fairseq.models.roberta import XLMRModel
from nlplingo.oregon.event_models.uoregon.tools.utils import *
from nlplingo.oregon.event_models.uoregon.layers import DynamicLSTM, CRF, GCN, SelfAttention, Elmo
#from nlplingo.oregon.event_models.uoregon.models.pipeline._01.iterators import upos_map, deprel_map, ner_map
from nlplingo.oregon.nlplingo.tasks.sequence.generator import upos_map, ner_map, deprel_map


class ArgumentModel(nn.Module):
    def __init__(self, opt, label_map):
        super(ArgumentModel, self).__init__()
        print('========= argument_model.ArgumentModel.__init__ START ===========')
        """decode.bash
        upos_dim= 30
        dist_dim= 30
        self.rep_dim= 60
        deprel_dim= 30
        hidden_dim= 200
        self.rep_dim= 860
        use_ner= 0
        use_biw2v= 0
        num_last_layer_xlmr= 1
        self.xlmr_dim= 768
        xlmr_model_dir= models/xlmr.base
        dropout_xlmr= 0.1
        self.word_embed_dim= 768
        self.rep_dim= 1628
        hidden_dim= 200
        deprel_dim= 30
        self.word_embed_dim= 768
        """
        self.opt = opt
        print('Using {} for argument model...'.format(opt['xlmr_version']))

        print('upos_dim=', self.opt['upos_dim'])
        print('dist_dim=', self.opt['dist_dim'])

        self.upos_embedding = nn.Embedding(
            num_embeddings=len(upos_map),
            embedding_dim=self.opt['upos_dim'],
            padding_idx=0
        )

        self.dist_embedding = nn.Embedding(
            num_embeddings=NUM_DISTANCES,
            embedding_dim=self.opt['dist_dim'],
            padding_idx=0
        )

        self.rep_dim = self.opt['upos_dim'] + self.opt['dist_dim']
        print('self.rep_dim=', self.rep_dim)
        print('deprel_dim=', self.opt['deprel_dim'])
        print('hidden_dim=', self.opt['hidden_dim'])

        self.deprel_embedding = nn.Embedding(
            num_embeddings=len(deprel_map),	# <==
            # num_embeddings=len(DEPREL_TO_ID),	# ==>
            embedding_dim=self.opt['deprel_dim'],
            padding_idx=0
        )

        self.gcn_layer = GCN(
            in_dim=self.opt['deprel_dim'],
            hidden_dim=self.opt['hidden_dim'],
            num_layers=2,
            opt=opt
        )

        self.rep_dim += self.opt['hidden_dim'] * 4
        print('self.rep_dim=', self.rep_dim)
        print('use_ner=', self.opt['use_ner'])

        if self.opt['use_ner']:
            self.ner_embedding = nn.Embedding(
                num_embeddings=len(ner_map),
                embedding_dim=self.opt['ner_dim'],
                padding_idx=0
            )
            self.rep_dim += self.opt['ner_dim']
        # *********************************************
        print('use_biw2v=', self.opt['use_biw2v'])
        print('num_last_layer_xlmr=', self.opt['num_last_layer_xlmr'])
        if not self.opt['use_biw2v']:
            if 'base' in self.opt['xlmr_version']:
                self.xlmr_dim = 768 * self.opt['num_last_layer_xlmr']
            elif 'large' in self.opt['xlmr_version']:
                self.xlmr_dim = 1024 * self.opt['num_last_layer_xlmr']
            print('self.xlmr_dim=', self.xlmr_dim)
            print('xlmr_model_dir=', self.opt['xlmr_model_dir'])
            print('dropout_xlmr=', self.opt['dropout_xlmr'])

            self.xlmr_embedding = XLMRModel.from_pretrained(
                #os.path.join(WORKING_DIR, 'tools', 'xlmr_resources', self.opt['xlmr_version']),	# <==
                self.opt['xlmr_model_dir'],								# ==>
                checkpoint_file='model.pt')
            self.dropout = nn.Dropout(self.opt['dropout_xlmr'])
            self.word_embed_dim = self.xlmr_dim
            print('self.word_embed_dim=', self.word_embed_dim)
        else:
            self.biw2v_embedding = nn.Embedding(
                opt['biw2v_size'],
                embedding_dim=300,
                padding_idx=PAD_ID
            )
            self.load_pretrained_biw2v()
            if not self.opt['finetune_biw2v']:
                self.biw2v_embedding.weight.requires_grad = False
            self.word_embed_dim = 300

        self.rep_dim += self.word_embed_dim
        print('self.rep_dim=', self.rep_dim)
        # ********************************************
        self.self_att = SelfAttention(self.rep_dim, opt)
        print('hidden_dim=', self.opt['hidden_dim'])
        self.fc_argument = nn.Sequential(
            nn.Linear(self.rep_dim + self.word_embed_dim, self.opt['hidden_dim']), 
            nn.ReLU(),
            #nn.Linear(self.opt['hidden_dim'], len(ARGUMENT_TAG_MAP))	# <==
            #nn.Linear(self.opt['hidden_dim'], 5)			# ==>	TODO
            nn.Linear(self.opt['hidden_dim'], len(label_map))
        )
        # ************ ACL idea **********************
        print('deprel_dim=', self.opt['deprel_dim'])
        print('self.word_embed_dim=', self.word_embed_dim)
        self.fc_edge = nn.Linear(self.word_embed_dim * 2, self.opt['deprel_dim'])
        self.edge_loss_func = nn.CrossEntropyLoss(reduction='none')
        print('========= argument_model.ArgumentModel.__init__ END ===========')

    def load_pretrained_biw2v(self):
        embed = self.biw2v_embedding
        vecs = self.opt['biw2v_vecs']
        pretrained = torch.from_numpy(vecs)
        embed.weight.data.copy_(pretrained)

    def get_xlmr_reps(self, inputs):
        print('============ argument_model.get_xlmr_reps START =============')
        """
        xlmr_ids.shape= torch.Size([30, 57])
        retrieve_ids.shape= torch.Size([30, 33])
        type(all_hiddens)= <class 'list'>
        len(all_hiddens)= 13
        all_hiddens[0].shape= torch.Size([30, 57, 768])
        all_hiddens[1].shape= torch.Size([30, 57, 768])
        all_hiddens[-1].shape= torch.Size([30, 57, 768])
        batch_size= 30
        len(all_hiddens)= 12
        self.opt['num_last_layer_xlmr']= 1
        used_layers= [11]
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        retrieved_reps.shape= torch.Size([33, 768])
        token_reps.shape= torch.Size([30, 33, 768])
        """
        xlmr_ids = inputs[0]
        retrieve_ids = inputs[2]
        print('xlmr_ids.shape=', xlmr_ids.shape)
        print('retrieve_ids.shape=', retrieve_ids.shape)

        # all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
        all_hiddens = self.xlmr_embedding.extract_features(xlmr_ids, return_all_hiddens=True)
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
            print('retrieved_reps.shape=', retrieved_reps.shape)
            token_reps.append(retrieved_reps)

        token_reps = torch.stack(token_reps, dim=0)  # [batch size, original seq len, xlmr_dim x num_layers]
        print('token_reps.shape=', token_reps.shape)
        print('============ argument_model.get_xlmr_reps END =============')
        return token_reps

    # def supervise_deprel_embeds(self, h, inputs):
    #     xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights, triggers, entity_tags, eid, pad_masks = inputs
    #     batch_size, seq_len, rep_dim = h.shape
    #     edge_reps = get_edge_reps(h, head_ids, pad_masks,
    #                               self.opt[
    #                                   'device'])  # [batch size, seq len, xlmr dim]
    #     edge_reps = edge_reps.repeat(1, 1, len(deprel_map)).view(batch_size, seq_len, len(deprel_map), -1)  # [batch size, seq len, num deps, xlmr dim]	# <==
    #     #edge_reps = edge_reps.repeat(1, 1, len(DEPREL_TO_ID)).view(batch_size, seq_len, len(DEPREL_TO_ID), -1)  # [batch size, seq len, num deps, xlmr dim]	# ==> TODO
    #     edge_reps = self.fc_edge(edge_reps)
    #
    #     dep_reps = self.deprel_embedding.weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1,
    #                                                                              1)  # [batch size, seq len, num deps, dep dim]
    #     # ****** take dot product to compute similarity scores **********
    #     logits = (edge_reps * dep_reps).sum(dim=3)  # [batch size, seq len, num deps]
    #     logits = logits.transpose(1, 2)  # [batch size, num deps, seq len]
    #     targets = deprel_ids  # [batch size, seq len]
    #     edge_loss = self.edge_loss_func(logits, targets)  # [batch size, seq len]
    #
    #     input_masks = pad_masks.long().eq(0).float()  # [batch size, seq len]
    #     num_edges = torch.sum(input_masks)
    #
    #     supervise_loss = torch.sum(edge_loss * input_masks) / num_edges
    #     return supervise_loss

    def forward(self, inputs):
        print('============= argument_model.forward START ============')
        xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, lang_weights, triggers, entity_tags, eid, pad_masks = inputs
        print('xlmr_ids.shape=', xlmr_ids.shape)
        print('biw2v_ids.shape=', biw2v_ids.shape)
        print('retrieve_ids.shape=', retrieve_ids.shape)
        print('upos_ids.shape=', upos_ids.shape)
        print('xpos_ids.shape=', xpos_ids.shape)
        print('head_ids.shape=', head_ids.shape)
        print('deprel_ids.shape=', deprel_ids.shape)
        print('ner_ids.shape=', ner_ids.shape)
        print('lang_weights.shape=', lang_weights.shape)
        print('trigger.shape=', triggers.shape)
        print('entity_tags.shape=', entity_tags.shape)
        print('eid.shape=', eid.shape)
        print('pad_masks.shape=', pad_masks.shape)

        """
        xlmr_ids.shape= torch.Size([16, 64])
        biw2v_ids.shape= torch.Size([16, 44])
        retrieve_ids.shape= torch.Size([16, 44])
        upos_ids.shape= torch.Size([16, 44])
        xpos_ids.shape= torch.Size([16, 44])
        head_ids.shape= torch.Size([16, 44])
        deprel_ids.shape= torch.Size([16, 44])
        ner_ids.shape= torch.Size([16, 44])
        lang_weights.shape= torch.Size([16])
        trigger.shape= torch.Size([16])
        entity_tags.shape= torch.Size([16, 44])
        eid.shape= torch.Size([16])
        pad_masks.shape= torch.Size([16, 44])
        batch_size= 16 seq_len= 44
        upos_reps.shape= torch.Size([16, 44, 30])
        dist_reps.shape= torch.Size([16, 44, 30])
        self.opt[use_biw2v]= 0
        """

        batch_size, seq_len = pad_masks.shape
        print('batch_size=', batch_size, 'seq_len=', seq_len)

        upos_reps = self.upos_embedding(upos_ids)  # [batch size, seq len, upos dim]
        print('upos_reps.shape=', upos_reps.shape)
        dist_reps = get_dist_embeds(triggers, batch_size, seq_len, self.dist_embedding, self.opt['device'])
        print('dist_reps.shape=', dist_reps.shape)

        word_feats = []
        print('self.opt[use_biw2v]=', self.opt['use_biw2v'])
        if not self.opt['use_biw2v']:
            """From below self.get_xlmr_reps()
            xlmr_ids.shape= torch.Size([16, 64])
            retrieve_ids.shape= torch.Size([16, 44])
            type(all_hiddens)= <class 'list'>
            len(all_hiddens)= 13
            all_hiddens[0].shape= torch.Size([16, 64, 768])
            all_hiddens[1].shape= torch.Size([16, 64, 768])
            all_hiddens[-1].shape= torch.Size([16, 64, 768])
            batch_size= 16
            len(all_hiddens)= 12
            self.opt['num_last_layer_xlmr']= 1
            used_layers= [11]
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            retrieved_reps.shape= torch.Size([44, 768])
            token_reps.shape= torch.Size([16, 44, 768])            
            """
            word_embeds = self.get_xlmr_reps(inputs)  # [batch size, seq len, xlmr dim]

            word_embeds = self.dropout(word_embeds)
        else:
            print('** using biw2v')
            word_embeds = self.biw2v_embedding(biw2v_ids)
        print('word_embeds.shape=', word_embeds.shape)

        """
        word_embeds.shape= torch.Size([16, 44, 768])
        word_deprel_reps.shape= torch.Size([16, 44, 30])
        adj.shape= torch.Size([16, 44, 44])
        word_deprel_reps.shape= torch.Size([16, 44, 200])
        trigger_deprel_reps.shape= torch.Size([16, 44, 200])
        word_reps.shape= torch.Size([16, 44, 1628])
        trigger_reps.shape= torch.Size([16, 44, 1628])
        word_reps.shape= torch.Size([16, 44, 1628])
        """

        # ********** supervise deprel embeds **********
        # edge_loss = self.supervise_deprel_embeds(word_embeds, inputs)
        edge_loss = 0
        # *********************************************

        word_feats.append(word_embeds)
        word_feats.append(upos_reps)
        word_feats.append(dist_reps)

        word_deprel_reps = self.deprel_embedding(deprel_ids)
        print('word_deprel_reps.shape=', word_deprel_reps.shape)
        adj = get_full_adj(head_ids, retrieve_ids, self.opt['device'])
        print('adj.shape=', adj.shape)
        word_deprel_reps, _ = self.gcn_layer(word_deprel_reps, adj)
        print('word_deprel_reps.shape=', word_deprel_reps.shape)
        trigger_deprel_reps = get_trigger_reps(word_deprel_reps, triggers).unsqueeze(1).repeat(1, seq_len, 1)  # [batch size, sep len, xlmr dim]
        print('trigger_deprel_reps.shape=', trigger_deprel_reps.shape)

        word_feats.append(word_deprel_reps)
        word_feats.append(trigger_deprel_reps)
        word_feats.append(torch.abs(trigger_deprel_reps - word_deprel_reps))
        word_feats.append(trigger_deprel_reps * word_deprel_reps)

        if self.opt['use_ner']:
            ner_reps = self.ner_embedding(ner_ids)
            word_feats.append(ner_reps)

        word_reps = torch.cat(word_feats, dim=2)
        print('word_reps.shape=', word_reps.shape)
        trigger_reps = get_trigger_reps(word_reps, triggers).unsqueeze(1).repeat(1, seq_len, 1)  # [batch size, sep len, xlmr dim]
        print('trigger_reps.shape=', trigger_reps.shape)

        word_reps = trigger_reps * word_reps
        print('word_reps.shape=', word_reps.shape)

        """From below self.self_att()
        input_masks.shape= torch.Size([16, 44])
        slf_attn_mask.shape= torch.Size([16, 44, 44])
        non_pad_mask.shape= torch.Size([16, 44, 1])
        enc_output.shape= torch.Size([16, 44, 1628])
        position_embed_for_satt= 1
        position_ids.shape= torch.Size([16, 44])
        enc_output.shape= torch.Size([16, 44, 1628])
        """
        word_reps, _ = self.self_att(word_reps, pad_masks)
        print('word_reps.shape=', word_reps.shape)

        """
        word_reps.shape= torch.Size([16, 44, 1628])
        word_reps.shape= torch.Size([16, 44, 2396])
        raw_scores.shape= torch.Size([16, 44, 12])
        token_masks.shape= torch.Size([16, 44])
        cl_loss= tensor(2.4789, device='cuda:0', grad_fn=<DivBackward0>)
        edge_loss= 0
        probs.shape= torch.Size([16, 44, 12])
        preds.shape= torch.Size([16, 44])
        loss= tensor(2.2310, device='cuda:0', grad_fn=<AddBackward0>)
        """

        word_reps = torch.cat(
            [word_reps, word_embeds],
            dim=2
        )
        print('word_reps.shape=', word_reps.shape)

        raw_scores = self.fc_argument(word_reps)  # [batch size, seq len, num tags]
        print('raw_scores.shape=', raw_scores.shape)

        token_masks = pad_masks.eq(0).float()
        print('token_masks.shape=', token_masks.shape)

        cl_loss, probs, preds = compute_batch_loss(raw_scores, entity_tags, token_masks, instance_weights=lang_weights)
        print('cl_loss=', cl_loss)
        print('edge_loss=', edge_loss)
        print('probs.shape=', probs.shape)
        print('preds.shape=', preds.shape)

        loss = self.opt['edge_lambda'] * edge_loss + (1 - self.opt['edge_lambda']) * cl_loss
        print('loss=', loss)
        print('============= argument_model.forward END ============')
        return loss, preds

    def predict(self, combined_task_inputs):
        print('============= argument_model.predict START =========')
        xlmr_ids, biw2v_ids, retrieve_ids, upos_ids, xpos_ids, head_ids, deprel_ids, ner_ids, triggers, eid, pad_masks = combined_task_inputs

        # print('xlmr_ids.shape=', xlmr_ids.shape)		[30, 57]
        # print('biw2v_ids.shape=', biw2v_ids.shape)		[30, 33]
        # print('retrieve_ids.shape=', retrieve_ids.shape)	[30, 33]
        # print('upos_ids.shape=', upos_ids.shape)		[30, 33]
        # print('xpos_ids.shape=', xpos_ids.shape)		[30, 33]
        # print('head_ids.shape=', head_ids.shape)		[30, 33]
        # print('deprel_ids.shape=', deprel_ids.shape)		[30, 33]
        # print('ner_ids.shape=', ner_ids.shape)		[30, 33]
        # print('triggers.shape=', triggers.shape)		[30]
        # print('eid.shape=', eid.shape)			[30]
        # print('pad_masks.shape=', pad_masks.shape)		[30, 33]

        batch_size, seq_len = pad_masks.shape

        upos_reps = self.upos_embedding(upos_ids)  # [batch size, seq len, upos dim]
        dist_reps = get_dist_embeds(triggers, batch_size, seq_len, self.dist_embedding, self.opt['device'])

        # print('upos_reps.shape=', upos_reps.shape)			[30, 33, 30]
        # print('dist_reps.shape=', dist_reps.shape)			[30, 33, 30]
        # print("self.opt['use_biw2v']=", self.opt['use_biw2v'])	0	

        word_feats = []
        if not self.opt['use_biw2v']:
            word_embeds = self.get_xlmr_reps(combined_task_inputs)  # [batch size, seq len, xlmr dim]

            word_embeds = self.dropout(word_embeds)
        else:
            word_embeds = self.biw2v_embedding(biw2v_ids)
        # print('word_embeds.shape=', word_embeds.shape)	[30, 33, 768]

        word_feats.append(word_embeds)
        word_feats.append(upos_reps)
        word_feats.append(dist_reps)

        word_deprel_reps = self.deprel_embedding(deprel_ids)
        adj = get_full_adj(head_ids, retrieve_ids, self.opt['device'])
        word_deprel_reps, _ = self.gcn_layer(word_deprel_reps, adj)
        trigger_deprel_reps = get_trigger_reps(word_deprel_reps, triggers).unsqueeze(1).repeat(1, seq_len, 1)  # [batch size, sep len, xlmr dim]

        # print('word_deprel_reps.shape=', word_deprel_reps.shape)		[30, 33, 200]
        # print('adj.shape=', adj.shape)					[30, 33, 33]
        # print('word_deprel_reps.shape=', word_deprel_reps.shape)		[30, 33, 200]
        # print('trigger_deprel_reps.shape=', trigger_deprel_reps.shape)	[30, 33, 200]

        word_feats.append(word_deprel_reps)
        word_feats.append(trigger_deprel_reps)
        word_feats.append(torch.abs(trigger_deprel_reps - word_deprel_reps))
        word_feats.append(trigger_deprel_reps * word_deprel_reps)

        # print("self.opt['use_ner']=", self.opt['use_ner'])	0
        if self.opt['use_ner']:
            ner_reps = self.ner_embedding(ner_ids)
            word_feats.append(ner_reps)

        word_reps = torch.cat(word_feats, dim=2)
        # print('word_reps.shape=', word_reps.shape)		[30, 33, 1628]

        trigger_reps = get_trigger_reps(word_reps, triggers).unsqueeze(1).repeat(1, seq_len, 1)  # [batch size, sep len, xlmr dim]
        # print('trigger_reps.shape=', trigger_reps.shape)	[30, 33, 1628]

        word_reps = trigger_reps * word_reps
        # print('word_reps.shape=', word_reps.shape)		[30, 33, 1628]

        """ When call self.self_att() below
        input_masks.shape= torch.Size([30, 33])
        slf_attn_mask.shape= torch.Size([30, 33, 33])
        non_pad_mask.shape= torch.Size([30, 33, 1])
        enc_output.shape= torch.Size([30, 33, 1628])
        position_embed_for_satt= 1
        position_ids.shape= torch.Size([30, 33])
        enc_output.shape= torch.Size([30, 33, 1628])
        """
        word_reps, _ = self.self_att(word_reps, pad_masks)
        # print('word_reps.shape=', word_reps.shape)		[30, 33, 1628]

        word_reps = torch.cat(
            [word_reps, word_embeds],
            dim=2
        )
        # print('word_reps.shape=', word_reps.shape)		[30, 33, 2396]

        raw_scores = self.fc_argument(word_reps)  # [batch size, seq len, num tags]
        # print('raw_scores.shape=', raw_scores.shape)		[30, 33, 12]

        token_masks = pad_masks.eq(0).float()
        entity_preds = torch.argmax(raw_scores, dim=2).long() * token_masks.long()
        # print('token_masks.shape=', token_masks.shape)	[30, 33]
        # print('entity_preds.shape=', entity_preds.shape)	[30, 33]

        probs = torch.softmax(raw_scores, dim=2)  # [batch size, seq len, num classes]
        # print('probs.shape=', probs.shape)			[30, 33, 12]

        """
            entity_preds= tensor([[ 3,  3,  4,  5,  5,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,
                      3,  3,  3,  3,  3,  3,  3,  6,  7,  7,  7,  3,  3,  3,  3],
                    [ 3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  5,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  6,  7,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  6,  7,  7,  7,  7,  7,  3,  0,  0],
                    [ 3,  3,  3,  3,  3,  6,  7,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  5,  5,  5,  5,  5,  5,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  3,  3,  3,  3,  3,  3,  3,  6,
                      7,  3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  4,  3,  3,  3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  4,  5,  3,  3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  6,  7,  3,  3,  3,  3,  3,
                      3,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      6,  3,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  6,  7,  3,  3,  3,  3,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  6,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  3,  3,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  5,  5,  5,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  5,  5,  5,  3,  3,  3,  6,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  8, 11, 11, 11, 11, 11,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  8, 11, 11, 11, 11, 11,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  5,  5,  5,  5,  5,  3,  3,  3,  3,  3,  3,  3,  3,  6,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  3,  5,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  6,
                      7,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  4,  3,  3,  3,  3,  3,  6,  6,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [ 3,  3,  3,  3,  3,  4,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
                      3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
                   device='cuda:0')
            probs= tensor([[[1.1168e-12, 2.5829e-12, 6.1225e-12,  ..., 5.6271e-09,
                      7.6096e-08, 8.8167e-08],
                     [8.9221e-13, 2.2523e-12, 5.5693e-12,  ..., 5.1459e-09,
                      5.5401e-08, 5.5402e-08],
                     [4.9063e-09, 4.8340e-09, 1.9968e-09,  ..., 1.6185e-05,
                      2.2680e-04, 2.8433e-05],
                     ...,
                     [8.4289e-13, 2.5055e-12, 6.1558e-12,  ..., 4.0031e-09,
                      5.3314e-08, 7.3456e-08],
                     [6.1697e-12, 1.4442e-11, 3.8848e-11,  ..., 1.6425e-08,
                      3.9797e-07, 1.0860e-06],
                     [1.0555e-12, 2.4275e-12, 5.7865e-12,  ..., 5.0906e-09,
                      7.4863e-08, 1.0263e-07]],
            
                    [[1.3705e-12, 2.5311e-12, 6.1232e-12,  ..., 1.3011e-08,
                      8.8079e-08, 2.2395e-07],
                     [1.2803e-12, 2.6615e-12, 6.6574e-12,  ..., 1.5308e-08,
                      7.7462e-08, 1.2939e-07],
                     [1.8388e-12, 3.6695e-12, 8.3890e-12,  ..., 2.1921e-08,
                      1.1262e-07, 1.6080e-07],
                     ...,
                     [1.0013e-12, 2.1821e-12, 6.0111e-12,  ..., 8.8796e-09,
                      6.4776e-08, 1.7182e-07],
                     [2.3456e-12, 4.3797e-12, 1.2402e-11,  ..., 1.6345e-08,
                      1.3718e-07, 4.6316e-07],
                     [1.3735e-12, 2.4432e-12, 6.1151e-12,  ..., 1.2238e-08,
                      9.3299e-08, 2.8162e-07]],
            
                    [[5.2102e-14, 9.0607e-14, 1.7959e-13,  ..., 1.0838e-09,
                      3.2204e-09, 1.3049e-08],
                     [4.4066e-14, 8.4767e-14, 1.6680e-13,  ..., 1.1195e-09,
                      2.5025e-09, 7.5451e-09],
                     [5.7656e-14, 1.1395e-13, 2.1501e-13,  ..., 1.4185e-09,
                      3.0650e-09, 7.2904e-09],
                     ...,
                     [8.0382e-14, 1.9794e-13, 4.8370e-13,  ..., 1.2600e-09,
                      4.1817e-09, 1.3255e-08],
                     [2.8645e-12, 3.1558e-12, 7.4554e-12,  ..., 1.2975e-08,
                      1.8952e-07, 2.6335e-06],
                     [7.8002e-14, 1.2314e-13, 2.4968e-13,  ..., 1.3987e-09,
                      4.9060e-09, 2.3695e-08]],
            
                    ...,
            
                    [[9.6163e-14, 1.9982e-13, 3.9772e-13,  ..., 3.2113e-09,
                      5.2772e-09, 1.4878e-08],
                     [1.2484e-11, 1.7996e-11, 1.9031e-11,  ..., 4.3214e-07,
                      8.1072e-07, 6.5115e-07],
                     [2.6778e-11, 2.2870e-11, 3.3277e-11,  ..., 2.5126e-07,
                      2.4906e-06, 1.9328e-05],
                     ...,
                     [3.7306e-02, 4.2389e-02, 3.6747e-02,  ..., 6.5021e-02,
                      6.5860e-02, 7.8383e-02],
                     [3.7306e-02, 4.2389e-02, 3.6747e-02,  ..., 6.5021e-02,
                      6.5860e-02, 7.8383e-02],
                     [3.7306e-02, 4.2389e-02, 3.6747e-02,  ..., 6.5021e-02,
                      6.5860e-02, 7.8383e-02]],
            
                    [[3.0249e-14, 7.0043e-14, 1.4678e-13,  ..., 4.8590e-10,
                      1.9103e-09, 5.7838e-09],
                     [2.4591e-10, 1.6229e-10, 5.7729e-11,  ..., 1.5785e-06,
                      1.1090e-05, 2.7864e-06],
                     [5.8275e-13, 7.5668e-13, 1.2182e-12,  ..., 5.4186e-09,
                      5.6736e-08, 3.0347e-07],
                     ...,
                     [3.8307e-02, 4.3550e-02, 3.7779e-02,  ..., 6.5669e-02,
                      6.6611e-02, 7.9756e-02],
                     [3.8307e-02, 4.3550e-02, 3.7779e-02,  ..., 6.5669e-02,
                      6.6611e-02, 7.9756e-02],
                     [3.8307e-02, 4.3550e-02, 3.7779e-02,  ..., 6.5669e-02,
                      6.6611e-02, 7.9756e-02]],
            
                    [[1.1015e-12, 2.6133e-12, 6.7187e-12,  ..., 8.8687e-09,
                      6.7803e-08, 4.9128e-08],
                     [9.4971e-12, 1.8937e-11, 3.3766e-11,  ..., 9.4450e-08,
                      6.8199e-07, 3.0716e-07],
                     [3.3704e-12, 5.4446e-12, 1.3414e-11,  ..., 1.8952e-08,
                      3.1152e-07, 6.1141e-07],
                     ...,
                     [3.2763e-02, 3.8756e-02, 3.3914e-02,  ..., 6.3191e-02,
                      6.1866e-02, 7.1638e-02],
                     [3.2763e-02, 3.8756e-02, 3.3914e-02,  ..., 6.3191e-02,
                      6.1866e-02, 7.1638e-02],
                     [3.2763e-02, 3.8756e-02, 3.3914e-02,  ..., 6.3191e-02,
                      6.1866e-02, 7.1638e-02]]], device='cuda:0')
            token_masks= tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                   device='cuda:0')
        """
        print('============= argument_model.predict END =========')
        return entity_preds, probs, token_masks


def pool_naive_path(reps, triggers, device):
    triggers = triggers.long().data.cpu().numpy()
    batch_size, seq_len, rep_dim = reps.shape
    pooled = []
    for b_id in range(batch_size):
        sentence = []
        for j in range(seq_len):
            with torch.no_grad():
                min_id = min(j, triggers[b_id])
                max_id = max(j, triggers[b_id])
                positions = set(range(min_id, max_id + 1))
                pool_mask = [0 if k in positions else 1 for k in range(seq_len)]
                pool_mask = torch.Tensor(pool_mask).bool().to(device)
            token_rep = max_pooling2d(pool_mask, reps[b_id])
            sentence.append(token_rep)
        sentence = torch.stack(sentence, dim=0)  # [seq len, rep dim]
        pooled.append(sentence)
    pooled = torch.stack(pooled, dim=0).to(device)  # [batch size, seq len, rep dim]
    return pooled


def get_dist_embeds(triggers, batch_size, seq_len, embed_layer, device):
    padding_id = 0
    out_of_left_id = 1
    out_of_right_id = 2
    with torch.no_grad():
        positions = torch.from_numpy(np.array(range(seq_len)))
        batch_position = positions.expand(batch_size, seq_len)
        batch_distance = batch_position - triggers.unsqueeze(
            1).long().cpu() + WINDOW_SIZE + 3  # 3 = 1 for padding, 1 for out of leftcontext, 1 for out of rightcontext

        out_of_left_mask = (batch_distance < 3).long()
        out_of_right_mask = (batch_distance >= NUM_DISTANCES).long()

        batch_outofleft = torch.ones_like(batch_distance).long() * out_of_left_mask * out_of_left_id
        batch_outofright = torch.ones_like(batch_distance).long() * out_of_right_mask * out_of_right_id

        out_of_context_mask = out_of_left_mask + out_of_right_mask
        out_of_context_mask = (out_of_context_mask != 1).long()
        batch_distance = batch_distance * out_of_context_mask + batch_outofleft + batch_outofright
        batch_distance = batch_distance.to(device)

    dist_reps = embed_layer(batch_distance)  # [batch size, seq len, dist dim]
    return dist_reps


def get_trigger_reps(batch_reps, triggers):
    '''
    :param batch_reps: [batch size, sequence length, rep dim]
    :param triggers: [batch size, ]
    :return: reps of anchor words: [batch size, rep dim]
    '''
    batch_reps = batch_reps.clone()
    ids = triggers.view(-1, 1).long()
    ids = ids.expand(ids.size(0), batch_reps.size(2)).unsqueeze(1)

    trigger_reps = torch.gather(batch_reps, 1, ids)
    trigger_reps = trigger_reps.squeeze(1)
    return trigger_reps
