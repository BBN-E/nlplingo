#from python.clever.event_models.uoregon.models.pipeline._01.modules.argument_model import ArgumentModel
#from python.clever.event_models.uoregon.models.pipeline._01.iterators import ArgumentIterator, get_arguments
#from python.clever.event_models.uoregon.models.pipeline._01.local_constants import *
from nlplingo.oregon.event_models.uoregon.tools.utils import *
import torch
#from nlplingo.oregon.event_models.uoregon.define_opt import logger
from datetime import datetime

from nlplingo.oregon.nlplingo.tasks.sequence.argument_model import ArgumentModel
from nlplingo.oregon.nlplingo.tasks.sequence.generator import ArgumentGenerator, get_arguments


class ArgumentTrainer(object):
    def __init__(self, opt, logger, biw2v_map, eval_mode=False, train_examples=None, train_docs=None, dev_examples=None, dev_docs=None, label_map=None):
        """
        :type opt: dict()
        :type train_examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type train_docs: list[nlplingo.text.text_theory.Document]
        :type dev_examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type dev_docs: list[nlplingo.text.text_theory.Document]
        :type label_map: dict[str, int]
        """
        print('label_map=', label_map)

        self.opt = opt
        self.logger = logger
        self.model = ArgumentModel(opt, label_map)
        self.model.to(self.opt['device'])
        #self.id2tag = dict([(v, k) for k, v in ARGUMENT_TAG_MAP.items()])
        self.id2tag = dict([(v, k) for k, v in label_map.items()])

        if not self.opt['finetune_xlmr']:
            for name, param in self.model.named_parameters():
                if 'xlmr' in name:
                    param.requires_grad = False
        elif self.opt['finetune_on_arb']:  # only update xlmr embeddings
            for name, param in self.model.named_parameters():
                found_in_list = False
                for finetuned_layer in opt['finetuned_xlmr_layers']:
                    if name.startswith(finetuned_layer):
                        found_in_list = True
                        break
                if not found_in_list:
                    param.requires_grad = False
        self._print_args()

        if not eval_mode:
            self.train_iterator = ArgumentGenerator(opt, self.model.xlmr_embedding, train_examples, train_docs, label_map, biw2v_map, is_eval_data=False)
            print('Argument model : Training data: {}'.format(self.train_iterator.num_examples))

            self.dev_iterator = ArgumentGenerator(opt, self.model.xlmr_embedding, dev_examples, dev_docs, label_map, biw2v_map, is_eval_data=True)
            print('Argument model : Dev data: {}'.format(self.dev_iterator.num_examples))

        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.Adam(self.parameters, betas=(0.9, 0.98), lr=opt['lr'])

        self.best_result = {
            'epoch': 0,
            'en_p': 0,
            'en_r': 0,
            'en_f1': 0
        }

        # print to screen opt
        cout = '*' * 100 + '\n'
        cout += 'Opt:\n'
        for arg in sorted(opt):
            if arg not in ['biw2v_vecs']:
                cout += '{}: {}\n'.format(arg, opt[arg])
        cout += '*' * 100 + '\n'
        print(cout)
        self.logger.info(cout)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('-' * 100)
        self.logger.info('> trainable params: Argument model')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.info('>>> {0}: {1}'.format(name, param.shape))
        self.logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('-' * 100)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        for b_id, batch in enumerate(self.train_iterator):
            batch = move_to_device(batch, self.opt['device'])
            loss, preds = self.model(batch)

            loss.backward()
            if self.opt['grad_clip_xlmr']:
                params_for_clip = self.model.parameters()
            else:
                params_for_clip = [param for name, param in self.model.named_parameters() if
                                   not name.startswith('xlmr_embedding')]
            torch.nn.utils.clip_grad_norm_(params_for_clip, self.opt['max_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()

            if b_id % 20 == 0:
                print(
                    '{}: step {}/{} (epoch {}/{}), loss = {:.3f}'.format(
                        datetime.now(), b_id, self.train_iterator.__len__(), epoch, self.opt['num_epoch'],
                        loss.item()
                    ))
                self.logger.info(
                    '{}: step {}/{} (epoch {}/{}), loss = {:.3f}'.format(
                        datetime.now(), b_id, self.train_iterator.__len__(), epoch, self.opt['num_epoch'],
                        loss.item()
                    ))

        self.eval(epoch)
        #if not self.opt['save_last_epoch'] and not self.opt['delete_nonbest_ckpts']:
        if self.opt['save_every_epoch']:
            print('Saving model...')
            self.save_model(epoch)
        self.train_iterator.shuffle_batches()
        print('=' * 50)
        self.logger.info('=' * 50)

    def prepare_training(self):
        if self.opt['train_strategy'].startswith('retrain'):
            if self.opt['initialize_with_pretrained']:
                epoch = self.load_saved_model()
                self.opt['argument_eval_epoch'] = epoch
                self.eval(self.opt['argument_eval_epoch'])

            # # *********** initialize pretrained model ***********
            # if self.opt['initialize_with_pretrained']:
            #     ensure_dir('checkpoints')
            #     argument_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
            #                              len(re.findall(
            #                                  r'argument[.]epoch[-]\d+[.]config',
            #                                  fname)) > 0]
            #     if len(argument_config_paths) > 0:
            #         argument_config_path = argument_config_paths[0]
            #         self.opt['argument_eval_epoch'] = int(
            #             re.findall(r'epoch[-](\d+)[.]config',
            #                        argument_config_path)[0])
            #
            #         print('Initializing model with pretrained model at checkpoint: {}'.format(
            #             self.opt['argument_eval_epoch']))
            #         logger.info('Initializing model with pretrained model at checkpoint: {}'.format(
            #             self.opt['argument_eval_epoch']))
            #         self.load_saved_model()
            #         self.eval(self.opt['argument_eval_epoch'])
            # else:
            #     # *********** English checkpoints *************
            #     existing_ckpts = [fname for fname in get_files_in_dir('checkpoints') if
            #                       os.path.basename(fname).startswith('argument')]
            #     print('Deleting all existing pretrained models!\nStart training from scratch...')
            #     logger.info('Deleting all existing pretrained models!\nStart training from scratch...')
            #     for ckpt in existing_ckpts:
            #         os.remove(ckpt)
            return 0
        elif self.opt['train_strategy'].startswith('cont_train'):
            epoch = self.load_saved_model()
            self.opt['argument_eval_epoch'] = epoch
            return self.opt['argument_eval_epoch'] + 1

            # argument_config_paths = [fname for fname in get_files_in_dir('checkpoints') if
            #                          len(re.findall(
            #                              r'argument[.]epoch[-]\d+[.]config',
            #                              fname)) > 0]
            # if len(argument_config_paths) > 0:
            #     argument_config_path = argument_config_paths[0]
            #     self.opt['argument_eval_epoch'] = int(
            #         re.findall(r'epoch[-](\d+)[.]config',
            #                    argument_config_path)[0])
            #     self.load_saved_model()
            #     print('Continue training at checkpoint: {}'.format(self.opt['argument_eval_epoch']))
            #     logger.info('Continue training at checkpoint: {}'.format(self.opt['argument_eval_epoch']))
            #     return self.opt['argument_eval_epoch'] + 1
            # else:
            #     print(
            #         'Saved model from hidden_eval.py is not found.\nStart training from scrach.'.format(
            #             self.opt['train_strategy']))
            #     logger.info('Saved model from hidden_eval.py is not found.\nStart training from scrach.'.format(
            #         self.opt['train_strategy']))
            #     return 0

    def train(self):
        print('Start training: Argument model')
        start_epoch = self.prepare_training()
        for epoch in range(start_epoch, start_epoch + self.opt['num_epoch']):
            self.train_epoch(epoch)
        #out = '**************************************************\nEnd of training: Argument model\nBest epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f})'.format(
        #    self.best_result['epoch'],
        #    self.best_result['en_p'],
        #    self.best_result['en_r'],
        #    self.best_result['en_f1']
        #)
        if not self.opt['save_every_epoch']:
            print('Saving model...')
            self.save_model(epoch)
            return -1, -1

        # if not self.opt['save_last_epoch']:
        #     print(out)
        #     self.logger.info(out)
        #     return self.best_result['epoch'], self.best_result['en_f1']
        # else:
        #     print('Saving model...')
        #     self.save_model(epoch)
        #     return -1, -1

    def compute_TP_FP_FN(self, ps, ls, ids, id2ori):
        batch_size = ls.shape[0]
        ori_examples = [id2ori.get(ids[k]) for k in range(batch_size)]

        TP_FN, TP_FP, TP = 0, 0, 0
        for b_id in range(batch_size):
            gold_agents, gold_patients, _, _ = get_arguments(self.id2tag, ls[b_id], ori_examples[b_id],
                                                             seperate_outputs=True)
            pred_agents, pred_patients, _, _ = get_arguments(self.id2tag, ps[b_id], ori_examples[b_id],
                                                             seperate_outputs=True)

            gold_agents = set(gold_agents)
            gold_patients = set(gold_patients)

            pred_agents = set(pred_agents)
            pred_patients = set(pred_patients)

            TP_FN += len(gold_agents) + len(gold_patients)
            TP_FP += len(pred_agents) + len(pred_patients)

            TP += np.sum(
                [1 for pred_agent in pred_agents if pred_agent in gold_agents]
                + [1 for pred_patient in pred_patients if pred_patient in gold_patients]
            )   # TODO this is just checking for equality in text span, does not consider if they are the same token indices

        return (TP, TP_FP, TP_FN)

    def record_scores(self, preds, batch, id2ori_example, log=False):
        with torch.no_grad():
            lang_weights = batch[-5]
            labels = batch[-3]
            eids = batch[-2].long().data.cpu().numpy().tolist()
            lang_ws = lang_weights.data.cpu().numpy().tolist()
            en_indices = [k for k in range(len(lang_ws)) if lang_ws[k] == 1.]
            ar_indices = [k for k in range(len(lang_ws)) if lang_ws[k] < 1.]

            en_preds, en_labels, en_ids = preds[en_indices, :], labels[en_indices, :], [eids[k] for k in en_indices]
            ar_preds, ar_labels, ar_ids = preds[ar_indices, :], labels[ar_indices, :], [eids[k] for k in ar_indices]
            return {
                'english': self.compute_TP_FP_FN(en_preds, en_labels, en_ids, id2ori_example),
                'arabic': self.compute_TP_FP_FN(ar_preds, ar_labels, ar_ids, id2ori_example)
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
            _, preds = self.model(batch)

            score = self.record_scores(preds, batch, self.dev_iterator.id2ori_example, log=True)
            en_TP += score['english'][0]
            en_TP_FP += score['english'][1]
            en_TP_FN += score['english'][2]

            ###################################
        en_p, en_r, en_f1 = self.compute_scores(en_TP, en_TP_FP, en_TP_FN)
        out = '=' * 50 + '\n' + 'dev: p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
            en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
        )
        print(out)
        self.logger.info(out)
        # if not self.opt['save_last_epoch'] and en_f1 > self.best_result['en_f1']:
        #     self.best_result['epoch'] = epoch
        #     self.best_result['en_p'] = en_p
        #     self.best_result['en_r'] = en_r
        #     self.best_result['en_f1'] = en_f1
        #     print('-> New best epoch')
        #     self.logger.info('-> New best epoch')
        #     if self.opt['delete_nonbest_ckpts']:
        #         print('Saving model...')
        #         self.save_model(epoch)

        self.model.train()

    def eval_with_saved_model(self):
        saved_epoch = self.load_saved_model()
        self.model.eval()
        en_TP, en_TP_FP, en_TP_FN = 0, 0, 0
        for b_id, batch in enumerate(self.dev_iterator):
            batch = move_to_device(batch, self.opt['device'])
            _, preds = self.model(batch)

            score = self.record_scores(preds, batch, self.dev_iterator.id2ori_example, log=True)
            en_TP += score['english'][0]
            en_TP_FP += score['english'][1]
            en_TP_FN += score['english'][2]

            ###################################
        en_p, en_r, en_f1 = self.compute_scores(en_TP, en_TP_FP, en_TP_FN)

        print(
            'Argument model : Epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
                saved_epoch, en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
            ))

        self.logger.info(
            'Argument model : Epoch: {}\ndev: (p: {:.3f},r: {:.3f},f1 = {:.3f}), TP: {}, pred_P: {}, gold_P: {}'.format(
                saved_epoch, en_p, en_r, en_f1, en_TP, en_TP_FP, en_TP_FN
            )
        )

        return -1, en_f1

    def load_saved_model(self):
        model_file = os.path.join(self.opt['argument_model_dir'], 'argument.model')
        # ensure_dir('checkpoints')
        # model_file = os.path.join('checkpoints',
        #                           'argument-model.seed-{}.epoch-{}.saved'.format(
        #                               self.opt['seed'], self.opt['argument_eval_epoch']))

        epoch = 0
        print('Loading checkpoint: {}'.format(model_file))
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file)

            # fairseq xlmr weirdly change the parameter names -> handle this as below:
            if 'xlmr_embedding.model.encoder.sentence_encoder.embed_tokens.weight' in set(
                    self.model.state_dict().keys()) and 'xlmr_embedding.model.decoder.sentence_encoder.embed_tokens.weight' in set(
                    checkpoint['model'].keys()):
                converted_state_dict = {}
                for key in checkpoint['model']:
                    converted_key = key.replace('xlmr_embedding.model.decoder.', 'xlmr_embedding.model.encoder.')
                    converted_state_dict[converted_key] = checkpoint['model'][key]
                checkpoint['model'] = converted_state_dict
            elif 'xlmr_embedding.model.decoder.sentence_encoder.embed_tokens.weight' in set(
                    self.model.state_dict().keys()) and 'xlmr_embedding.model.encoder.sentence_encoder.embed_tokens.weight' in set(
                    checkpoint['model'].keys()):
                converted_state_dict = {}
                for key in checkpoint['model']:
                    converted_key = key.replace('xlmr_embedding.model.encoder.', 'xlmr_embedding.model.decoder.')
                    converted_state_dict[converted_key] = checkpoint['model'][key]
                checkpoint['model'] = converted_state_dict
            
            self.model.load_state_dict(checkpoint['model'])
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
            print('Loaded!')

            # try:
            #     checkpoint = torch.load(model_file)
            #     # fairseq xlmr weirdly change the parameter names -> handle this as below:
            #     if "xlmr_embedding.model.encoder.sentence_encoder.embed_tokens.weight" in set(
            #             self.model.state_dict().keys()):
            #         converted_state_dict = {}
            #         for key in checkpoint['model']:
            #             converted_key = key.replace('xlmr_embedding.model.decoder.', 'xlmr_embedding.model.encoder.')
            #             converted_state_dict[converted_key] = checkpoint['model'][key]
            #         checkpoint['model'] = converted_state_dict
            #
            #     self.model.load_state_dict(checkpoint['model'])
            #     if 'epoch' in checkpoint:
            #         epoch = checkpoint['epoch']
            #     print('Loaded!')
            # except BaseException:
            #     print("Cannot load model from {}".format(model_file))
            #     exit(1)
        else:
            print('No checkpoints found!')
            exit(1)

        return epoch

    def save_model(self, epoch):
        # ensure_dir('checkpoints')
        #
        # if self.opt['delete_nonbest_ckpts']:
        #     existing_ckpts = [fname for fname in get_files_in_dir('checkpoints') if
        #                       os.path.basename(fname).startswith('argument')]
        #     for ckpt in existing_ckpts:
        #         os.remove(ckpt)
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
            output_dir = os.path.join(self.opt['output_dir'], 'epoch-%d' % (epoch))
            ensure_dir(output_dir)

            ckpt_fpath = os.path.join(output_dir, 'argument.model')
            # ckpt_fpath = os.path.join('checkpoints', 'argument-model.seed-{}.epoch-{}.saved'.format(self.opt['seed'], epoch))
            torch.save(params, ckpt_fpath)
            print('... to: {}'.format(ckpt_fpath))

            opt['argument_model_dir'] = output_dir
            write_json(opt, write_path=os.path.join(output_dir, 'argument.config'))
            # write_json(opt, write_path=os.path.join('checkpoints', 'argument.epoch-{}.config'.format(epoch)))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def predict(self, combined_task_inputs):
        with torch.no_grad():
            argument_preds, argument_probs, _ = self.model.predict(combined_task_inputs)
            return argument_preds, argument_probs
