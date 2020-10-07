import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceREDecodeLoader, SentenceRETrainLoader
from .utils import AverageMeter
from datetime import datetime
import time
logger = logging.getLogger()
import numpy as np
import codecs
import glob
import re

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, f1):
        """
        Higher f1 is better. This is why the count is incremented when score < self.best_score
        :param val_loss:
        :return:
        """
        score = f1

        if self.best_score is None:
            self.best_score = score
            self.print_best_val_loss(f1)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_best_val_loss(f1)
            self.counter = 0

    ### model IO
    def print_best_val_loss(self, val_loss):
        if self.verbose:
            print(f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss



class SentenceRETrain(nn.Module):
    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 extractor_params,
                 hyper_params,
                 features,
                 event_domain
                 ):

        super().__init__()
        self.hyper_params = hyper_params
        self.extractor_params = extractor_params
        self.features = features
        self.optimize_params = extractor_params['optimizer']

        self.test_path = test_path

        self.max_epoch = self.hyper_params.epoch
        # Load data
        batch_size = self.hyper_params.batch_size

        self.features = features
        self.event_domain = event_domain

        # Load data
        if train_path != None:
            if hyper_params.mention_pool:
                self.train_loader = SentenceRETrainLoader(
                    train_path,
                    model.rel2id,
                    model.sentence_encoder.tokenize,
                    batch_size,
                    True,
                    event_domain,
                    mention_pool=True)
            else:
                self.train_loader = SentenceRETrainLoader(
                    train_path,
                    model.rel2id,
                    model.sentence_encoder.tokenize,
                    batch_size,
                    True,
                    event_domain)

        if val_path != None:
            if hyper_params.mention_pool:
                self.val_loader = SentenceRETrainLoader(
                    val_path,
                    model.rel2id,
                    model.sentence_encoder.tokenize,
                    batch_size,
                    False,
                    event_domain,
                    mention_pool=True)
            else:
                self.val_loader = SentenceRETrainLoader(
                    val_path,
                    model.rel2id,
                    model.sentence_encoder.tokenize,
                    batch_size,
                    False,
                    event_domain)

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        lr = self.optimize_params['lr']
        self.lr = lr
        weight_decay = self.optimize_params['weight_decay']
        opt = self.optimize_params['name']
        warmup_step = self.optimize_params['warmup_step']
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, self.lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, self.lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        self.ckpt = self.hyper_params.save_model_path + '/'  # + timestamp + '/'
        if not os.path.exists(self.ckpt):
            os.makedirs(self.ckpt, exist_ok=True)
        # log_dir = self.hyper_params.log_dir
        # if not os.path.exists(log_dir):
        #    os.mkdir(log_dir)
        # self.log_path = log_dir + '/{}.log'.format(now_ckpt)

    def train_model(self, metric='micro_f1'):
        best_metric = 0
        global_step = 0

        if hasattr(self.hyper_params, 'patience'):
            early_stopping = EarlyStopping(patience=self.hyper_params.patience, verbose=True)

        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1)  # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            if hasattr(self.hyper_params, 'patience'):
                early_stopping(result[metric])
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                # timestamp = datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt + 'epoch_' + str(epoch))
                best_metric = result[metric]

            if hasattr(self.hyper_params, 'patience'):
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break

        logging.info("Best %s on val set: %f" % (metric, best_metric))

        if self.test_path is not None:
            # emit best val score to dev.score_file
            with codecs.open(self.hyper_params.dev_score_file, 'w', encoding='utf-8') as f:
                f.write(str(best_metric) + '\n')
            self.test_model()

    def test_model(self, metric='micro_f1'):
        logging.info("Testing...")
        files = glob.glob(self.ckpt + "/epoch_*")

        file_nums = []
        for i, s in enumerate(files):
            num_str = re.search("epoch_(\d+)$", files[i])  # capture only integer before ".csv" and EOL
            file_nums.append(int(num_str.group(1)))  # convert to number
        new_number = str(max(file_nums))

        final_model_path = ""
        for file in files:
            if new_number in os.path.basename(file):
                final_model_path = file
                break
        self.model.load_state_dict(torch.load(final_model_path)['state_dict'])

        if self.hyper_params.mention_pool:
            test_loader = SentenceRETrainLoader(
                self.test_path,
                self.model.rel2id,
                self.model.sentence_encoder.tokenize,
                self.hyper_params.batch_size,
                False,
                self.event_domain,
                mention_pool=True)
        else:
            test_loader = SentenceRETrainLoader(
                self.test_path,
                self.model.rel2id,
                self.model.sentence_encoder.tokenize,
                self.hyper_paramsbatch_size,
                False,
                self.event_domain)

        result = self.eval_model(test_loader)
        with codecs.open(self.hyper_params.test_score_file, 'w', encoding='utf-8') as f:
            f.write(str(result[metric]) + '\n')
            f.write(final_model_path + '\n')

    def eval_model(self, eval_loader, return_preds_only=False):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1)  # (B)
                # Save result
                for i in range(pred.size(0)):
                    if not return_preds_only:
                        pred_result.append(pred[i].item())
                    else:
                        pred_result.append((self.id2rel[pred[i].item()], score[i].item()))
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)

        if not return_preds_only:
            result = eval_loader.dataset.eval(pred_result, self.extractor_params['model_type'])
            return result
        else:
            return pred_result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class SentenceREDecode(nn.Module):
    def __init__(self,
                 model,
                 decode_triplets,
                 extractor_params,
                 hyper_params,
                 rel2id, optimize_batches=False):

        super().__init__()

        self.hyper_params = hyper_params
        self.extractor_params = extractor_params

        # Load data
        batch_size = self.hyper_params.batch_size

        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        self.model.eval()

        if not optimize_batches:
            logging.info('Total candidate relations given to Sentence Decoder: %s', len(decode_triplets))
        else:
            logging.info('Total unique sentences given to Sentence Decoder: %s', len(decode_triplets))
            logging.info('Total relations given to Sentence Decoder: %s', sum([len(decode_triplet[1]) for decode_triplet in decode_triplets]))

        if not optimize_batches:
            self.test_loader = SentenceREDecodeLoader(hyper_params,
                                                      decode_triplets,
                                                      model.rel2id,
                                                      model.sentence_encoder.tokenize,
                                                      batch_size,
                                                      shuffle=False,
                                                      optimize_batches=False
                                                      )
        else:
            start = time.time()
            tokenized_data = []
            start_examples = 0
            real_size = 0
            chunk_range_dict = {}
            self.delete_indices = set()

            actual_sent_ct = 0
            for idx, decode_triplet in enumerate(decode_triplets):
                if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
                    # We delete indices containing invalid head/tail pairs (beyond GLOBAL_MAX_SENT_LENGTH)
                    tokenized_output, del_indices = model.sentence_encoder.tokenize_mention_pool_decode(decode_triplet)
                    for del_index in del_indices:
                        self.delete_indices.add(start_examples + del_index)

                    start_examples += len(decode_triplet[1])
                    num_real_examples = len(tokenized_output[2])
                    if num_real_examples > 0:
                        chunk_range_dict[actual_sent_ct] = list(range(real_size, real_size + num_real_examples))
                        tokenized_data.append(tokenized_output)
                        actual_sent_ct += 1
                        real_size += num_real_examples
                else:
                    tokenized_output = model.sentence_encoder.tokenize(decode_triplet, blank_padding=False)
                    tokenized_data.append(tokenized_output)


            num_original_relations = sum([len(decode_triplet[2]) for decode_triplet in decode_triplets])
            logging.info('Deleted: %s', len(self.delete_indices))
            assert(real_size + len(self.delete_indices) == num_original_relations) # an invariant to ensure we deleted the right number of relations

            # optimize_batches should represent the 'sentence-length' optimization
            if self.hyper_params.encoder == 'bert_mention' or self.hyper_params.encoder == 'cnn_opt':
                final_order = []

                # sort so that the prediction order matches the original input order
                chunk_sort_order = sorted(range(len(tokenized_data)), key=tokenized_data.__getitem__)
                for chunk in chunk_sort_order:
                    final_order.extend(chunk_range_dict[chunk])

                assert(len(final_order) == real_size)
                self.sort_order = final_order
            else:
                # Below code was used an earlier operating on non-mention pool architectures
                self.sort_order = sorted(range(len(tokenized_data)), key=tokenized_data.__getitem__)

            tokenized_data.sort()
            end = time.time()
            logging.info('Tokenization seconds: %s', end - start)
            self.test_loader = SentenceREDecodeLoader(hyper_params,
                                                      tokenized_data,
                                                      model.rel2id,
                                                      model.sentence_encoder.tokenize,
                                                      batch_size,
                                                      shuffle=False,
                                                      optimize_batches=True
                                                      )

    def eval_model(self):
        eval_loader = self.test_loader
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = eval_loader
            # t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass

                args = data
                #print('args print')
                #print(data[0][0].shape)
                #print(data[0][1].shape)
                #print(len(args[1]))
                #print(len(args[2]))
                #if not self.hyper_params.mention_pool:
                #    logits = self.parallel_model(*args)
                #else:
                logits = self.model(*args)
                logits = nn.Softmax(-1)(logits)
                # logits = self.parallel_model.softmax(logits)
                score, pred = logits.max(-1)  # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append((self.id2rel[pred[i].item()], score[i].item()))
        return pred_result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
