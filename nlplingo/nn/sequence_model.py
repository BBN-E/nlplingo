import os
import logging
import random
import json

import numpy as np
from nlplingo.tasks.sequence.utils import print_scores
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torch

from transformers import AdamW, AutoConfig, AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup, XLMRobertaModel

from nlplingo.nn.extraction_model import ExtractionModel

logger = logging.getLogger(__name__)


class SequenceModel(ExtractionModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features=None):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        super(SequenceModel, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.optimizer_params = self.optimizer  # because self.optimizer from ExtractionModel is actually just optimization parameters when we are doing pytorch

        self.args = dict()
        # args['data_dir'] = params['data_dir']
        self.args['model_type'] = extractor_params['model_type']  # str. Model type selected from MODEL_CLASSES
        #self.args['model_name_or_path'] = extractor_params['extractor_name']  # str. Path to pre-trained model or shortcut name selected from ALL_MODELS
        self.args['model_name_or_path'] = 'xlm-roberta-base'
        self.args['output_dir'] = params['output_dir']  # str. The output directory where the model predictions and checkpoints will be written TODO this needs to be optional during decoding
        # args['label_filepath'] = params['labels']              # str. Path to a file containing all labels. If not specified, CoNLL-2003 labels are used
        self.args['config_name'] = ''  # str. Pretrained config name or path if not the same as model_name
        self.args['tokenizer_name'] = ''  # str. Pretrained tokenizer name or path if not the same as model_name
        self.args['cache_dir'] = extractor_params['cache_dir']  # str. Where do you want to store the pre-trained models downloaded from s3
        #self.args['max_seq_length'] = 256  # int. The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded
        self.args['do_train'] = True  # whether to run training
        self.args['do_eval'] = False  # whether to run eval on dev set
        self.args['do_predict'] = False  # whether to run predictions on the test set
        self.args['evaluate_during_training'] = False  # whether to run evaluation during training at each logging step

        self.args['do_lower_case'] = False  # Set this flag if you are using an uncased model
        self.args['keep_accents'] = False  # Set this flag if model is trained with accents
        self.args['strip_accents'] = False  # Set this flag if model is trained with accents
        self.args['use_fast'] = False  # Set this flag to use fast tokenization

        #self.args['per_gpu_train_batch_size'] = extractor_params.get('batch_size', 8)  # Batch size per GPU/CPU for training

        self.args['per_gpu_eval_batch_size'] = 8  # Batch size per GPU/CPU for evaluation
        self.args['gradient_accumulation_steps'] = 1  # Number of updates steps to accumulate before performing a backward/update pass
        #self.args['learning_rate'] = 5e-5  # The initial learning rate for Adam
        #self.args['learning_rate'] = extractor_params.get('lr', 5e-05)
        self.args['weight_decay'] = 0.0  # Weight decay if we apply some
        self.args['adam_epsilon'] = 1e-8  # Epsilon for Adam optimizer
        self.args['max_grad_norm'] = 1.0  # Max gradient norm
        #self.args['num_train_epochs'] = extractor_params.get('epoch', 4)  # Total number of training epochs to perform

        self.args['max_steps'] = -1  # If > 0: set total number of training steps to perform. Override num_train_epochs
        self.args['warmup_steps'] = 0  # Linear warmup over warmup_steps
        self.args['logging_steps'] = 0  # Log every X updates steps
        self.args['save_steps'] = 0  # Save checkpoint every X updates steps
        self.args['eval_all_checkpoints'] = False  # Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number
        self.args['no_cuda'] = False  # Avoid using CUDA when available
        self.args['overwrite_output_dir'] = True  # Overwrite the content of the output directory
        self.args['overwrite_cache'] = False  # Overwrite the cached training and evaluation sets
        self.args['seed'] = extractor_params['seed']  # random seed for initialization
        self.args['fp16'] = False  # Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
        self.args['fp16_opt_level'] = 'O1'  # For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html
        self.args['local_rank'] = -1  # For distributed training: local_rank
        self.args['server_ip'] = ''  # For distant debugging
        self.args['server_port'] = ''  # For distant debugging
        self.args['save_every_epoch'] = extractor_params.get('save_every_epoch', False)
        # tokenizer_args = {'do_lower_case': args['do_lower_case'], 'keep_accents': args['keep_accents'], 'strip_accents': args['strip_accents'], 'use_fast': args['use_fast']}
        #tokenizer_args = {'do_lower_case': self.args['do_lower_case']}

        # if (
        #     os.path.exists(self.args['output_dir']) and os.listdir(self.args['output_dir']) and self.args['do_train'] and not self.args['overwrite_output_dir']
        # ):
        #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(self.args['output_dir']))

        # Setup distant debugging if needed
        # if self.args['server_ip'] and self.args['server_port']:
        #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        #     import ptvsd
        #
        #     print("Waiting for debugger attach")
        #     ptvsd.enable_attach(address=(self.args['server_ip'], self.args['server_port']), redirect_output=True)
        #     ptvsd.wait_for_attach()

        # Setup CUDA, GPU & distributed training
        # if self.args['local_rank'] == -1 or self.args['no_cuda']:
        #     device = torch.device("cuda" if torch.cuda.is_available() and not self.args['no_cuda'] else "cpu")
        #     self.args['n_gpu'] = 0 if self.args['no_cuda'] else torch.cuda.device_count()
        # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #     torch.cuda.set_device(self.args['local_rank'])
        #     device = torch.device("cuda", self.args['local_rank'])
        #     torch.distributed.init_process_group(backend="nccl")
        #     self.args['n_gpu'] = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args['n_gpu'] = torch.cuda.device_count()
        self.args['device'] = device

        # Setup logging
        # logging.basicConfig(
        #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        #     datefmt="%m/%d/%Y %H:%M:%S",
        #     level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
        # )
        # logger.warning(
        #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        #     local_rank, device, n_gpu, bool(local_rank != -1), fp16,
        # )

        # Set seed
        self.set_seed(self.args)

        # Prepare CONLL-2003 task
        # labels = get_labels(args['label_filepath'])
        #print('labels:', labels)
        #num_labels = len(labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index

        # Load pretrained model and tokenizer
        # if self.args['local_rank'] not in [-1, 0]:
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.args['model_type'] = self.args['model_type'].lower()
        # 762ddd751172e9d3229e5da17a459eee6c0dfdc237c718944d0b1a85f06c7e1e.9ba214636e460976b286b4ce15e95d778f32439e9fdd8ddae7e3784f3a7e24a2

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        # logger.info("Training/evaluation parameters %s", args)

        """
        # Training
        if args['do_train']:
            train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, datalines,
                                                    mode="train")
            global_step, tr_loss = self.train(self.args, train_dataset, model, tokenizer, labels, pad_token_label_id)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if self.args['do_train'] and (self.args['local_rank'] == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(self.args['output_dir']) and self.args['local_rank'] in [-1, 0]:
                os.makedirs(self.args['output_dir'])

            logger.info("Saving model checkpoint to %s", self.args['output_dir'])
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.args['output_dir'])
            tokenizer.save_pretrained(self.args['output_dir'])

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(self.args['output_dir'], "training_args.bin"))
        """

    def train(self, examples, dev_examples=None):
        """ Train the model
        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type dev_examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        """
        all_input_ids = torch.tensor([e.input_ids for e in examples], dtype=torch.long)
        all_input_mask = torch.tensor([e.input_mask for e in examples], dtype=torch.long)
        all_segment_ids = torch.tensor([e.segment_ids for e in examples], dtype=torch.long)
        all_label_ids = torch.tensor([e.label_ids for e in examples], dtype=torch.long)
        train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        #self.args['train_batch_size'] = self.args['per_gpu_train_batch_size'] * max(1, self.args['n_gpu'])
        self.args['train_batch_size'] = self.args['per_gpu_train_batch_size']
        #print("**** Hyper-parameters self.args['train_batch_size']={}".format(str(self.args['train_batch_size'])))    # TODO

        #train_sampler = RandomSampler(train_dataset) if self.args['local_rank'] == -1 else DistributedSampler(train_dataset)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args['train_batch_size'])

        if self.args['max_steps'] > 0:
            t_total = self.args['max_steps']
            self.args['num_train_epochs'] = self.args['max_steps'] // (
            len(train_dataloader) // self.args['gradient_accumulation_steps']) + 1
        else:
            t_total = len(train_dataloader) // self.args['gradient_accumulation_steps'] * self.args['num_train_epochs']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args['weight_decay'],
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        #optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['learning_rate'], eps=self.args['adam_epsilon'])
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.optimizer_params['lr'], eps=self.args['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args['warmup_steps'], num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args['model_name_or_path'], "optimizer.pt")) and os.path.isfile(
                os.path.join(self.args['model_name_or_path'], "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.args['model_name_or_path'], "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args['model_name_or_path'], "scheduler.pt")))

        # if self.args['fp16']:
        #     try:
        #         from apex import amp
        #     except ImportError:
        #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #     self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args['fp16_opt_level'])

        # multi-gpu training (should be after apex fp16 initialization)
        #print("**** Hyper-parameters self.args['n_gpu']={}".format(str(self.args['n_gpu'])))
        if self.args['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        # if self.args['local_rank'] != -1:
        #     self.model = torch.nn.parallel.DistributedDataParallel(
        #         self.model, device_ids=[self.args['local_rank']], output_device=self.args['local_rank'], find_unused_parameters=True
        #     )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args['num_train_epochs'])
        logger.info("  n_gpu = %d", self.args['n_gpu'])
        logger.info("  train_batch_size = %d", self.args['train_batch_size'])
        logger.info("  Instantaneous batch size per GPU = %d", self.args['per_gpu_train_batch_size'])
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args['train_batch_size']
            * self.args['gradient_accumulation_steps']
            * (torch.distributed.get_world_size() if self.args['local_rank'] != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args['gradient_accumulation_steps'])
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        # if os.path.exists(self.args['model_name_or_path']):
        #     # set global_step to gobal_step of last saved checkpoint from model path
        #     try:
        #         global_step = int(self.args['model_name_or_path'].split("-")[-1].split("/")[0])
        #     except ValueError:
        #         global_step = 0
        #     epochs_trained = global_step // (len(train_dataloader) // self.args['gradient_accumulation_steps'])
        #     steps_trained_in_current_epoch = global_step % (
        #     len(train_dataloader) // self.args['gradient_accumulation_steps'])
        #
        #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        #     logger.info("  Continuing training from epoch %d", epochs_trained)
        #     logger.info("  Continuing training from global step %d", global_step)
        #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.args['num_train_epochs']), desc="Epoch", disable=self.args['local_rank'] not in [-1, 0]
        )
        self.set_seed(self.args)  # Added here for reproductibility

        epoch_counter = 0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args['local_rank'] not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.args['device']) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

                # TODO the following is currently broken if you are using distilbert, bert, xlnet, because 'model_type' looks like 'sequence_xlmr-base
                if self.args['model_type'] != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.args['model_type'] in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids

                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.args['n_gpu'] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']

                # if self.args['fp16']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    # if self.args['fp16']:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args['max_grad_norm'])
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])

                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1

                    # if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    #     # Log metrics
                    #     if (
                    #         args['local_rank'] == -1 and args['evaluate_during_training']
                    #     ):  # Only evaluate when single GPU otherwise metrics may not average well
                    #         results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                    #         for key, value in results.items():
                    #             tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    #     tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    #     tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args['logging_steps'], global_step)
                    #     logging_loss = tr_loss

                    if self.args['local_rank'] in [-1, 0] and self.args['save_steps'] > 0 and global_step % self.args[
                        'save_steps'] == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(self.args['output_dir'], "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.args['max_steps'] > 0 and global_step > self.args['max_steps']:
                    epoch_iterator.close()
                    break
            if self.args['max_steps'] > 0 and global_step > self.args['max_steps']:
                train_iterator.close()
                break

            if dev_examples is not None:
                predictions = self.predict_sequence(dev_examples, self.event_domain)
                """list[list[str]]"""
                print_scores(dev_examples, predictions, self.event_domain.sequence_types)

            if self.args['save_every_epoch']:
                output_dir = os.path.join(self.args['output_dir'], "epoch-{}".format(str(epoch_counter)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                # Good practice: save your training arguments together with the trained model
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

            epoch_counter += 1
        # if args['local_rank'] in [-1, 0]:
        #     tb_writer.close()

        return global_step, tr_loss / global_step

    def save_model(self):
        # Create output directory if needed
        if not os.path.exists(self.args['output_dir']) and self.args['local_rank'] in [-1, 0]:
            os.makedirs(self.args['output_dir'])

        logger.info("Saving model checkpoint to %s", self.args['output_dir'])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.args['output_dir'])
        self.tokenizer.save_pretrained(self.args['output_dir'])

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args['output_dir'], "training_args.bin"))

    def _read_model_type(self, config_filepath):
        with open(config_filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        return datas['model_type']

    def _read_labels(self, config_filepath):
        """
        :rtype: list[str]
        """
        with open(config_filepath, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        labels = [k for k, v in sorted(datas['label2id'].items(), key=lambda item: item[1])]
        return labels

    def set_seed(self, args):
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        if args['n_gpu'] > 0:
            torch.cuda.manual_seed_all(args['seed'])

    def predict_sequence(self, examples, domain):
        """
        :type examples: list[nlplingo.tasks.sequence.example.SequenceExample]
        :type domain: nlplingo.tasks.event_domain.EventDomain
        """
        eval_batch_size = 8

        if len(examples) == 0:
            return []

        all_input_ids = torch.tensor([e.input_ids for e in examples], dtype=torch.long)
        all_input_mask = torch.tensor([e.input_mask for e in examples], dtype=torch.long)
        all_segment_ids = torch.tensor([e.segment_ids for e in examples], dtype=torch.long)
        all_label_ids = torch.tensor([e.label_ids for e in examples], dtype=torch.long)
        all_subword_to_token_indices = torch.tensor([e.subword_to_token_indices for e in examples], dtype=torch.long)
        all_seq_length = torch.tensor([e.seq_length for e in examples], dtype=torch.long)
        eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subword_to_token_indices, all_seq_length)

        # print('evaluate(), len(datalines)=', len(datalines))
        # for i, l in enumerate(datalines):
        #     print(i, ':', len(l.split('\n')))
        # print('datalines[0]=', datalines[0])
        #
        # args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
        # print('args["n_gpu"]=', args['n_gpu'])  # 1
        # print('args["eval_batch_size"]=', args['eval_batch_size'])  # 8
        # print('args["local_rank"]=', args['local_rank'])  # -1

        # Note that DistributedSampler samples randomly
        #eval_sampler = SequentialSampler(eval_dataset) if args['local_rank'] == -1 else DistributedSampler(eval_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        # if args['n_gpu'] > 1:
        #     model = torch.nn.DataParallel(model)

        # Eval!
        # logger.info("***** Running evaluation %s *****", prefix)
        # logger.info("  Num examples = %d", len(eval_dataset))  # 10
        # logger.info("  Batch size = %d", args['eval_batch_size'])  # 8
        # eval_loss = 0.0
        # nb_eval_steps = 0
        preds = None
        out_label_ids = None
        subword_to_token_indices = None
        seq_lengths = None
        self.model.eval()
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            batch = tuple(t.to(self.args['device']) for t in batch)
            # print('evaluate(), len(batch)=', len(batch))  # 4
            # print('evaluate(), type(batch[0])=', type(batch[0]))  # Tensor

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                # print('evaluate(), len(inputs["input_ids"])=', len(inputs['input_ids']))  # 8
                # print('evaluate(), len(inputs["attention_mask"])=', len(inputs['attention_mask']))  # 8
                # print('evaluate(), len(inputs["labels"])=', len(inputs['labels']))  # 8

                # TODO following is broken if you are using distilbert, bert, xlnet, because self.model_type will look like 'sequence_xlmr-base'
                if self.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = self.model(**inputs)
                # print('evaluate(), type(outputs)=', type(outputs))  # tuple
                tmp_eval_loss, logits = outputs[:2]     # logits: eval_batch_size X max_seq_length X len(self.labels)
                # print('evaluate(), type(tmp_eval_loss)=', type(tmp_eval_loss))  # Tensor
                # print('evaluate(), type(logits)=', type(logits))  # Tensor
                # print('evaluate(), logits.size()=',
                #       logits.size())  # torch.Size([8, 256, 25])	, 256=max_seq_length, 25=num#-labels

                # if args['n_gpu'] > 1:
                #     tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                # eval_loss += tmp_eval_loss.item()
                # print('evaluate(), eval_loss=', eval_loss)  # 0.321...
            # nb_eval_steps += 1
            if preds is None:
                # print('evaluate(), preds is None')
                # we can't directly convert any tensor requiring gradients to numpy arrays.
                # so we need to call .detach() first to remove the computational graph tracking.
                # .cpu is in case the tensor is on the GPU, in which case you need to move it back to the CPU to convert it to a tensor
                preds = logits.detach().cpu().numpy()
                # print('evaluate(), type(preds)=', type(preds))  # numpy.ndarray
                # print('evaluate(), preds.shape=', preds.shape)  # (8, 256, 25)	, 256=max_seq_length, 25=num#-labels
                # print('===== preds ====')
                # print(preds)
                """
                [[[ 9.80186164e-01  1.09039581e+00  7.98717663e-02 ... -3.52853864e-01
                   -3.02161416e-03  3.08335900e+00]
                  [ 1.09557652e+00  2.87680054e+00 -1.23010492e+00 ... -1.33469689e+00
                   -8.73669028e-01  1.26455867e+00]
                  [ 1.04975331e+00  2.87343812e+00 -1.36169112e+00 ... -1.58172524e+00
                   -9.25045907e-01  1.63650835e+00]
                  ...
                  [ 9.44989860e-01  1.18976617e+00  2.95494553e-02 ... -6.13705873e-01
                   -1.68917496e-02  3.97478414e+00]
                  [ 9.44989860e-01  1.18976617e+00  2.95494553e-02 ... -6.13705873e-01
                   -1.68917496e-02  3.97478414e+00]
                  [ 9.44989860e-01  1.18976617e+00  2.95494553e-02 ... -6.13705873e-01
                   -1.68917496e-02  3.97478414e+00]]
                """

                out_label_ids = inputs["labels"].detach().cpu().numpy()  # this is the gold annotation
                # print('evaluate(), out_label_ids.shape=', out_label_ids.shape)  # (8, 256)
                # print('==== out_label_ids ====')
                # print(out_label_ids)
                """
                [[-100   11 -100 ... -100 -100 -100]
                 [-100   24 -100 ... -100 -100 -100]
                 [-100   24   24 ... -100 -100 -100]
                 ...
                 [-100   24   24 ... -100 -100 -100]
                 [-100   24 -100 ... -100 -100 -100]
                 [-100   24   24 ... -100 -100 -100]]
                """

                subword_to_token_indices = batch[4].detach().cpu().numpy()
                seq_lengths = batch[5].detach().cpu().numpy()
            else:
                # print('evaluate(), preds is not None')
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # append predictions from each subsequent batch
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)  # append gold labels from subsequent batches
                subword_to_token_indices = np.append(subword_to_token_indices, batch[4].detach().cpu().numpy(), axis=0)
                seq_lengths = np.append(seq_lengths, batch[5].detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        # print('evaluate(), before np.argmax, preds.shape=', preds.shape)  # (10, 256, 25)
        preds = np.argmax(preds, axis=2)
        # print('evaluate(), after np.argmax, preds.shape=', preds.shape)  # (10, 256)

        #label_map = {i: label for i, label in enumerate(self.labels)}
        label_map = self.event_domain.sequence_types_inv

        # print('evaluate(), label_map=')  # {index} -> class-label-string
        # print(label_map)
        """
        {0: 'B-harmful.both', 1: 'B-harmful.material', ..., 24: 'O'}
        """

        # print('evaluate(), out_label_ids.shape=', out_label_ids.shape)  # (10, 256)
        #out_label_list = [[] for _ in range(out_label_ids.shape[0])]  # list of 10 empty lists ; don't really need this
        first_subword_preds_list = [[] for _ in range(out_label_ids.shape[0])]  # list of 10 empty lists
        all_subword_preds_list = [[] for _ in range(out_label_ids.shape[0])]

        # print('pad_token_label_id=', pad_token_label_id)  # -100
        #
        # print('out_label_ids[0]=', out_label_ids[0])
        """
        [-100   11 -100 -100   24 -100 -100   24   24 -100 -100   24 -100 -100
         -100   24   24 -100 -100 -100   24   24   24    7   24   24   24 -100
         -100   24 -100 -100   24 -100   24 -100   24 -100 -100   24   24   24
           24 -100   24 -100   24   24   24   24   24   24    3 -100   24   24
           24   24    8 -100 -100 ...
          ...
         -100 -100 -100 -100]
        """

        # print('preds[0]=', preds[0])
        """
        [24  1  1 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 11
         24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
         24 24 24 24  1  1 24 24 24 24  1 24 24 24 24 24 24 24 24 24 24 24 24 24
         ...
         24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24]
        """

        assert subword_to_token_indices.shape[0] == out_label_ids.shape[0]
        assert subword_to_token_indices.shape[1] == out_label_ids.shape[1]

        #print(subword_to_token_indices)

        # print('==== seq_lengths ====')
        # print(seq_lengths)
        #
        # print('==== out_label_ids ====')
        # print(out_label_ids)

        for i in range(out_label_ids.shape[0]):         # each example
            for j in range(seq_lengths[i]):
            #for j in range(out_label_ids.shape[1]):     # each time step    , TODO this can be improved by keeping track of number of subwords of each example
                if out_label_ids[i, j] != self.pad_token_label_id:  # use this to only use prediction from first subword of each token
                    #out_label_list[i].append(label_map[out_label_ids[i][j]])   # don't really need this
                    first_subword_preds_list[i].append(label_map[preds[i][j]])
                #if subword_to_token_indices[i, j] != self.pad_token_label_id:
                all_subword_preds_list[i].append(label_map[preds[i][j]])

        # print('==== subword_to_token_indices ====')
        # print(subword_to_token_indices)
        # print('==== all_subword_preds_list ====')
        # print('len=', len(all_subword_preds_list[0]), all_subword_preds_list)
        # print('==== first_subword_preds_list ====')
        # print('len=', len(first_subword_preds_list[0]), first_subword_preds_list)

        any_subword_preds_list = []
        for i in range(len(all_subword_preds_list)):                             # each example
            eg_preds = ['O'] * len(first_subword_preds_list[i]) # number of tokens for this example   TODO the 'O' needs to be changed to task specific None label
            for j in range(len(all_subword_preds_list[i])-1, -1, -1):
                if subword_to_token_indices[i, j] == self.pad_token_label_id:
                    continue
                token_index = subword_to_token_indices[i, j]
                assert token_index < len(eg_preds)
                if all_subword_preds_list[i][j] != 'O':
                    eg_preds[token_index] = all_subword_preds_list[i][j]
            any_subword_preds_list.append(eg_preds)

        # print('==== any_subword_preds_list ====')
        # print('len=', len(any_subword_preds_list[0]), any_subword_preds_list)

        # print('evaluate(), preds_list=')
        # for i, l in enumerate(preds_list):
        #     print(i, len(l), ':', l)
        """
            0 35 : ['B-harmful.material', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-neutral.verbal', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-harmful.material', 'O', 'O', 'O', 'O', 'B-harmful.material']
        """

        # print('evaluate(), out_label_list=')
        # for i, l in enumerate(out_label_list):
        #     print(i, len(l), ':', l)
        """
            0 35 : ['B-neutral.verbal', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-helpful.verbal', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-harmful.verbal', 'O', 'O', 'O', 'O', 'B-neutral.both']
        """

        # results = {
        #     "loss": eval_loss,
        #     # "precision": precision_score(out_label_list, preds_list),
        #     # "recall": recall_score(out_label_list, preds_list),
        #     # "f1": f1_score(out_label_list, preds_list),
        # }

        # logger.info("***** Eval results %s *****", prefix)
        # for key in sorted(results.keys()):
        #     logger.info("  %s = %s", key, str(results[key]))

        return any_subword_preds_list


class SequenceXLMRBase(SequenceModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features=None):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        super(SequenceXLMRBase, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        if hasattr(self.hyper_params, 'decode_mode') and self.hyper_params.decode_mode:       # decoding
            print('**** Decoding ****')
            # NOTE: read 'model_file' as model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_file)

            #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.args['device'])

            #self.model_type = self._read_model_type(os.path.join(self.model_file, 'config.json'))  # e.g. xlm-roberta
            #self.labels = self._read_labels(os.path.join(model_dir, 'config.json'))

            #self.pad_token_label_id = CrossEntropyLoss().ignore_index
        elif hasattr(self.hyper_params, 'continue_training') and self.hyper_params.continue_training:   # load a saved model for more training
            print('**** Continue Training ****')
            self.args['per_gpu_train_batch_size'] = self.hyper_params.batch_size  # Batch size per GPU/CPU for training
            self.args['num_train_epochs'] = self.hyper_params.epoch  # Total number of training epochs to perform

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_file)
            self.model.to(self.args['device'])
        else:                                   # training
            print('**** Training from Scratch ****')
            self.args['per_gpu_train_batch_size'] = self.hyper_params.batch_size  # Batch size per GPU/CPU for training
            self.args['num_train_epochs'] = self.hyper_params.epoch  # Total number of training epochs to perform
            num_labels = len(self.event_domain.sequence_types)

            config = AutoConfig.from_pretrained(
                self.args['config_name'] if self.args['config_name'] else self.args['model_name_or_path'],
                num_labels=num_labels,
                #id2label={str(i): label for i, label in enumerate(labels)},
                #label2id={label: i for i, label in enumerate(labels)},
                id2label={str(i): label for label, i in self.event_domain.sequence_types.items()},
                label2id={label: i for label, i in self.event_domain.sequence_types.items()},
                cache_dir=self.args['cache_dir'] if self.args['cache_dir'] else None,
            )
            # tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
            # logger.info("Tokenizer arguments: %s", tokenizer_args)

            tokenizer_args = {'do_lower_case': self.args['do_lower_case']}
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args['tokenizer_name'] if self.args['tokenizer_name'] else self.args['model_name_or_path'],
                # YS: xlm-roberta-base
                cache_dir=self.args['cache_dir'] if self.args['cache_dir'] else None,
                **tokenizer_args,
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.args['model_name_or_path'],
                from_tf=bool(".ckpt" in self.args['model_name_or_path']),  # YS: xlm-roberta-base
                config=config,
                cache_dir=self.args['cache_dir'] if self.args['cache_dir'] else None,
            )

            # if self.args['local_rank'] == 0:
            #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

            self.model.to(self.args['device'])


class SequenceXLMRCustom(SequenceModel):
    def __init__(self, params, extractor_params, event_domain, embeddings, hyper_params, features=None):
        """
        :type: params: dict
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        super(SequenceXLMRCustom, self).__init__(params, extractor_params, event_domain, embeddings, hyper_params, features)
        self.create_model()

    def create_model(self):
        if self.hyper_params.decode_mode:       # decoding
            self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            #self.tokenizer = AutoTokenizer.from_pretrained(self.model_file)
            self.model = XLMRLinearModel.from_pretrained(self.model_file, sorted(list(self.event_domain.sequence_types)))
            #self.model = AutoModelForTokenClassification.from_pretrained(self.model_file)
            self.model.to(self.args['device'])
        else:                                   # training
            self.args['per_gpu_train_batch_size'] = self.hyper_params.batch_size  # Batch size per GPU/CPU for training
            self.args['num_train_epochs'] = self.hyper_params.epoch  # Total number of training epochs to perform
            #num_labels = len(self.event_domain.sequence_types)

            tokenizer_args = {'do_lower_case': self.args['do_lower_case']}
            self.tokenizer = AutoTokenizer.from_pretrained(
                'xlm-roberta-base',
                cache_dir=self.args['cache_dir'] if self.args['cache_dir'] else None,
                **tokenizer_args,
            )

            self.model = XLMRLinearModel(sorted(list(self.event_domain.sequence_types)), self.args['cache_dir'])
            self.model.to(self.args['device'])

            self.model.config.save_pretrained(self.args['output_dir'])

            print(self.model.state_dict().keys())
            for k, v in self.model.state_dict().items():
                print(k, '\t\t\t', v.shape)


class XLMRLinearModel(torch.nn.Module):
    def __init__(self, labels, cache_dir):
        super(XLMRLinearModel, self).__init__()

        self.num_labels = len(labels)

        # load xlmr embedder
        self.config = AutoConfig.from_pretrained(
            'xlm-roberta-base',
            num_labels=self.num_labels,
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=cache_dir if cache_dir else None,
        )

        self.roberta = XLMRobertaModel(self.config)

        # projection layer
        self.classifier = torch.nn.Linear(in_features=768, out_features=self.num_labels)

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        #print(self.embedder.__dict__)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0] # (N,T,C) = (8, 256, 768)
        outputs = self.classifier(outputs) # (N,T,C) = (8, 256, 7)
        outputs_transposed = outputs.permute(0,2,1) # (N,C,T) = (8, 7, 256);    labels.shape = (8, 7) = (N,C)

        CELoss = torch.nn.CrossEntropyLoss()
        loss = CELoss(outputs_transposed, labels)
            # input: outputs of shape (N,C,T) = (8,7,256)
            # target: labels of shape (N,T) = (8,256)

        return loss, outputs

    def save_pretrained(self, save_directory):
        '''analogous to AutoConfig save_pretrained method'''
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), output_model_file)
        #transformers.modeling_utils.PreTrainedModel.save_pretrained(self, output_dir)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, labels, from_latest_checkpoint=False):
        '''model_path must have checkpoint file'''

        model = cls(labels, cache_dir=None)

        if from_latest_checkpoint: # automatically detect latest checkpoint file
            checkpoint_files = sorted([f for f in os.listdir(pretrained_model_path) if f.startswith('checkpoint-')], key=lambda f: int(f.replace('checkpoint-','')))

            if not checkpoint_files:
                raise Exception('no checkpoint file found in model dir {}'.format(pretrained_model_path))
            else:
                latest_checkpoint = checkpoint_files[-1]
                print("loading model from {}".format(latest_checkpoint))
                model_state_dict = torch.load(os.path.join(pretrained_model_path, latest_checkpoint, "pytorch_model.bin"))
                model.load_state_dict(model_state_dict)
        else:
            model_state_dict = torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin"))
            model.load_state_dict(model_state_dict)

        return model

