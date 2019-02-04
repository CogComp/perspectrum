import os
import random
from tqdm import tqdm, trange

import csv
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from experiment.bert.run_classifier import ColaProcessor, MrpcProcessor, logger, convert_examples_to_features, \
    set_optimizer_params_grad, copy_optimizer_params_to_model, accuracy, p_r_f1, tp_pcount_gcount
from experiment.bert.run_classifier import MnliProcessor


# Data Directory
FILE_PATH = {
    # Stance
    "stance_data": "../data/dataset/perspective_stances/",
    "stance_model_dir": "../model/stance",
    "stance_model_name": "bert_stance.pth",

    # Relevance
    "relevance_data": "../data/dataset/perspective_relevance/",
    "relevance_model_dir": "../model/relevance",
    "relevance_model_name": "bert_relevance.pth",

    # Evidence
    "evidence_data": "../data/dataset/perspective_evidence/",
    "evidence_model_dir": "../model/evidence",
    "evidence_model_name": "bert_evidence.pth",

    # Equivalence
    "equivalence_data": "../data/dataset/perspective_equivalence/",
    "equivalence_model_dir": "../model/equivalence",
    "equivalence_model_name": "bert_equivalence.pth",
}

# Default config file
DEFAULT_CONFIG = {
    "bert_model": "bert-base-uncased",
    "max_seq_length": 128,
    "do_lower_case": False,
    "train_batch_size": 32,
    "eval_batch_size": 8,
    "learning_rate": 5e-5,
    "num_train_epoch": 8,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128,
}

"""
TODO: Improvements
1. Look at where they cached the bert model, if it's under /home/ or /tmp, move it somewhere else
"""

class BertBaseline:
    def __init__(self, do_train=False, saved_model=None, **kwargs):
        """
        :param saved_model: path to trained model
        :param kwargs: see DEFAULT_CONFIG above
        """
        self._config = kwargs

        if self._config["local_rank"] == -1 or self._config["no_cuda"]:
            self._device = torch.device("cuda" if torch.cuda.is_available() and not self._config["no_cuda"] else "cpu")
            self._n_gpu = torch.cuda.device_count()
        else:
            self._device = torch.device("cuda", self._config["local_rank"])
            self._n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
            if self._config["fp16"]:
                logger.info("16-bits training currently not supported in distributed training")
                self._config["fp16"] = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device %s n_gpu %d distributed training %r", self._device, self._n_gpu, bool(self._config["local_rank"] != -1))

        if self._config["gradient_accumulation_steps"] < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self._config["gradient_accumulation_steps"]))

        train_batch_size = int(self._config["train_batch_size"] / self._config["gradient_accumulation_steps"])

        random.seed(self._config["seed"])
        np.random.seed(self._config["seed"])
        torch.manual_seed(self._config["seed"])
        if self._n_gpu > 0:
            torch.cuda.manual_seed_all(self._config["seed"])

        self._processor = MrpcProcessor()


        # Prepare model
        model = BertForSequenceClassification.from_pretrained(self._config["bert_model"],
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                  self._config["local_rank"]))
        if self._config["fp16"]:
            model.half()
        model.to(self._device)
        if self._config["local_rank"] != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[kwargs["local_rank"]],
                                                              output_device=kwargs["local_rank"])
        elif self._n_gpu > 1:
            model = torch.nn.DataParallel(model)

        self._model = model
        self._optimizer = None



    @property
    def config(self):
        """
        :return: The configurations of the model as a dict
        """
        return self._config

    # @property
    # def model(self):
    #     """
    #     :return: Underlying bert torch model
    #     """
    #     return self._model

    def train(self, train_data_dir, output_dir):

        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError("Output directory ({}) already exists and is not emp1ty.".format(output_dir))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Prepare optimizer
        if self._config["fp16"]:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                               for n, param in self._model.named_parameters()]
        elif self._config["optimize_on_cpu"]:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                               for n, param in self._model.named_parameters()]
        else:
            param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]

        train_examples = self._processor.get_train_examples(train_data_dir)
        num_train_steps = int(len(train_examples) / self._config["train_batch_size"] / self._config["gradient_accumulation_steps"] * self._config["num_train_epochs"])

        t_total = num_train_steps
        if self._config["local_rank"] != -1:
            t_total = t_total // torch.distributed.get_world_size()

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self._config["learning_rate"],
                             warmup=self._config["warmup_proportion"],
                             t_total=self._config["t_total"])

        label_list = self._processor.get_labels()

        tokenizer = BertTokenizer.from_pretrained(self._config["bert_model"], do_lower_case=self._config["do_lower_case"])

        train_features = convert_examples_to_features(
            train_examples, label_list, self._config["max_seq_length"], tokenizer)

        # Begin Training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self._config["train_batch_size"])
        logger.info("  Num steps = %d", self._config["num_train_steps"])
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if self._config["local_rank"] == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self._config["train_batch_size"])

        self._model.train()
        global_step = 0
        for _ in trange(int(self._config["num_train_epochs"]), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self._device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self._model(input_ids, segment_ids, input_mask, label_ids)
                if self._n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self._config["fp16"] and self._config["loss_scale"] != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * self._config["loss_scale"]
                if self._config["gradient_accumulation_steps"] > 1:
                    loss = loss / self._config["gradient_accumulation_steps"]
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self._config["gradient_accumulation_steps"] == 0:
                    if self._config["fp16"] or self._config["optimize_on_cpu"]:
                        if self._config["fp16"] and self._config["loss_scale"] != 1.0:
                            # scale down gradients for fp16 training
                            for param in self._model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / self.config["loss_scale"]
                        is_nan = set_optimizer_params_grad(param_optimizer, self._model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            loss_scale = loss_scale / 2
                            self._model.zero_grad()
                            continue
                        self.optimizer.step()
                        copy_optimizer_params_to_model(self._model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    self._model.zero_grad()
                    global_step += 1

        torch.save(self._model.state_dict(), output_dir + "output.pth")


    def _train_init(self, train_data_dir):
        """

        :param train_data_dir:
        :return:
        """
        # Prepare optimizer
        if self._config["fp16"]:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                               for n, param in self._model.named_parameters()]
        elif self._config["optimize_on_cpu"]:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                               for n, param in self._model.named_parameters()]
        else:
            param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]

        train_examples = self._processor.get_train_examples(train_data_dir)
        num_train_steps = int(len(train_examples) / self._config["train_batch_size"] / self._config["gradient_accumulation_steps"] * self._config["num_train_epochs"])

        t_total = num_train_steps
        if self._config["local_rank"] != -1:
            t_total = t_total // torch.distributed.get_world_size()

        self._optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self._config["learning_rate"],
                             warmup=self._config["warmup_proportion"],
                             t_total=self._config["t_total"])

        label_list = self._processor.get_labels()

        tokenizer = BertTokenizer.from_pretrained(self._config["bert_model"], do_lower_case=self._config["do_lower_case"])

        self._train_features = convert_examples_to_features(
            train_examples, label_list, self._config["max_seq_length"], tokenizer)


    def evaluate(self):
        pass


def train_and_test(data_dir, bert_model="bert-base-uncased", task_name=None,
                   output_dir=None, output_name="output.pth", max_seq_length=128, do_train=False, do_eval=False,
                   do_lower_case=False,train_batch_size=32, eval_batch_size=8, learning_rate=5e-5, num_train_epochs=10,
                   warmup_proportion=0.1,no_cuda=False, local_rank=-1, seed=42, gradient_accumulation_steps=1,
                   optimize_on_cpu=False, fp16=False, loss_scale=128, saved_model="", eval_dev_set=False):

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
    }

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if fp16:
            logger.info("16-bits training currently not supported in distributed training")
            fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(local_rank != -1))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if do_train:
        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError("Output directory ({}) already exists and is not emp1ty.".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_examples = None
    num_train_steps = None
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        dev_examples = processor.get_dev_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps * num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank))
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if do_train:
        train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
        dev_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        # Train features
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Dev features
        dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)

        if local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        dev_sampler = SequentialSampler(dev_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=eval_batch_size)

        model.train()

        for _epoch in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            tr_per_batch_loss = []
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if fp16 and loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * loss_scale
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                tr_per_batch_loss.append(loss.item())
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16 or optimize_on_cpu:
                        if fp16 and loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            loss_scale = loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

            # After every epoch, validate on dev
            eval_tp, eval_pred_c, eval_gold_c = 0, 0, 0
            eval_loss, eval_macro_p, eval_macro_r = 0, 0, 0

            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in dev_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # Micro F1 (aggregated tp, fp, fn counts across all examples)
                tmp_tp, tmp_pred_c, tmp_gold_c = tp_pcount_gcount(logits, label_ids)
                eval_tp += tmp_tp
                eval_pred_c += tmp_pred_c
                eval_gold_c += tmp_gold_c

                # Macro F1 (averaged P, R across mini batches)
                tmp_eval_p, tmp_eval_r, tmp_eval_f1 = p_r_f1(logits, label_ids)

                eval_macro_p += tmp_eval_p
                eval_macro_r += tmp_eval_r

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            # Micro F1 (aggregated tp, fp, fn counts across all examples)
            eval_micro_p = eval_tp / eval_pred_c
            eval_micro_r = eval_tp / eval_gold_c
            eval_micro_f1 = 2 * eval_micro_p * eval_micro_r / (eval_micro_p + eval_micro_r)

            # Macro F1 (averaged P, R across mini batches)
            eval_macro_p = eval_macro_p / nb_eval_steps
            eval_macro_r = eval_macro_r / nb_eval_steps
            eval_macro_f1 = 2 * eval_macro_p * eval_macro_r / (eval_macro_p + eval_macro_r)

            logger.info("Training Loss: {}".format(tr_loss))
            logger.info("Per mini batch training loss: {}".format(str(tr_per_batch_loss)))
            logger.info("Epoch {} Dev Macro Precision: {}".format(_epoch, eval_macro_p))
            logger.info("Epoch {} Dev Macro Recall: {}".format(_epoch, eval_macro_r))
            logger.info("Epoch {} Dev Macro F1: {}".format(_epoch, eval_macro_f1))

            torch.save(model.state_dict(), os.path.join(output_dir, "epoch_{}.pth".format(_epoch)))


    if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):

        if eval_dev_set:
            eval_examples = processor.get_dev_examples(data_dir)
        else:
            eval_examples = processor.get_test_examples(data_dir)

        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        model.load_state_dict(torch.load(saved_model))

        model.eval()
        # eval_loss, eval_accuracy = 0, 0

        eval_tp, eval_pred_c, eval_gold_c = 0, 0, 0
        eval_loss, eval_macro_p, eval_macro_r = 0, 0, 0

        raw_score = []

        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            # Micro F1 (aggregated tp, fp, fn counts across all examples)
            tmp_tp, tmp_pred_c, tmp_gold_c = tp_pcount_gcount(logits, label_ids)
            eval_tp += tmp_tp
            eval_pred_c += tmp_pred_c
            eval_gold_c += tmp_gold_c

            raw_score += zip(logits, label_ids)
            # Macro F1 (averaged P, R across mini batches)
            tmp_eval_p, tmp_eval_r, tmp_eval_f1 = p_r_f1(logits, label_ids)

            eval_macro_p += tmp_eval_p
            eval_macro_r += tmp_eval_r

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1


        # Micro F1 (aggregated tp, fp, fn counts across all examples)
        eval_micro_p = eval_tp / eval_pred_c
        eval_micro_r = eval_tp / eval_gold_c
        eval_micro_f1 = 2 * eval_micro_p * eval_micro_r / (eval_micro_p + eval_micro_r)

        # Macro F1 (averaged P, R across mini batches)
        eval_macro_p = eval_macro_p / nb_eval_steps
        eval_macro_r = eval_macro_r / nb_eval_steps
        eval_macro_f1 = 2 * eval_macro_p * eval_macro_r / (eval_macro_p + eval_macro_r)

        eval_loss = eval_loss / nb_eval_steps
        result = {'eval_loss': eval_loss,
                  'eval_micro_p': eval_micro_p,
                  'eval_micro_r': eval_micro_r,
                  'eval_micro_f1': eval_micro_f1,
                  # 'eval_macro_p': eval_macro_p,
                  # 'eval_macro_r': eval_macro_r,
                  # 'eval_macro_f1': eval_macro_f1,
                  'global_step': global_step,
                  # 'loss': tr_loss/nb_tr_steps
                  }

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        output_raw_score = os.path.join(output_dir, "raw_score.csv")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        with open(output_raw_score, 'w') as fout:
            fields = ["undermine_score", "support_score", "gold"]
            writer = csv.DictWriter(fout, fieldnames=fields)
            writer.writeheader()
            for score, gold in raw_score:
                writer.writerow({
                    "undermine_score": str(score[0]),
                    "support_score": str(score[1]),
                    "gold": str(gold)
                })



def stance_train(epoch=7):
    data_dir = FILE_PATH['stance_data']
    model_dir = FILE_PATH['stance_model_dir']
    model_name = FILE_PATH['stance_model_name']
    train_and_test(data_dir=data_dir, do_train=True, do_eval=False, output_dir=model_dir, output_name=model_name,
                   task_name="Mrpc", num_train_epochs=epoch)

def stance_test():
    data_dir = FILE_PATH['stance_data']
    model_dir = FILE_PATH['stance_model_dir']
    model_name = FILE_PATH['stance_model_name']
    model_path = os.path.join(model_dir, model_name)
    train_and_test(data_dir=data_dir, do_train=False, do_eval=True, output_dir=model_dir,task_name="Mrpc",
                   saved_model=model_path)


def relevance_train(epoch=7):
    data_dir = FILE_PATH['relevance_data']
    model_dir = FILE_PATH['relevance_model_dir']
    model_name = FILE_PATH['relevance_model_name']
    train_and_test(data_dir=data_dir, do_train=True, do_eval=False, output_dir=model_dir, output_name=model_name,
                   task_name="Mrpc", num_train_epochs=epoch)


def relevance_test():
    data_dir = FILE_PATH['relevance_data']
    model_dir = FILE_PATH['relevance_model_dir']
    model_name = FILE_PATH['relevance_model_name']
    model_path = os.path.join(model_dir, model_name)
    train_and_test(data_dir=data_dir, do_train=False, do_eval=True, output_dir=model_dir, task_name="Mrpc",
                  saved_model=model_path)

def equivalence_train():
    data_dir = FILE_PATH['equivalence_data']
    model_dir = FILE_PATH['equivalence_model_dir']
    model_name = FILE_PATH['equivalence_model_name']
    train_and_test(data_dir=data_dir, do_train=True, do_eval=False, output_dir=model_dir, output_name=model_name,
                   task_name="Mrpc")


def equivalence_test():
    data_dir = FILE_PATH['equivalence_data']
    model_dir = FILE_PATH['equivalence_model_dir']
    model_name = FILE_PATH['equivalence_model_name']
    model_path = os.path.join(model_dir, model_name)
    train_and_test(data_dir=data_dir, do_train=False, do_eval=True, output_dir=model_dir, task_name="Mrpc",
                  saved_model=model_path)


if __name__ == "__main__":
    stance_train()
    # stance_test()
    # relevance_train()
    # relevance_test()
    # equivalence_train()
    # equivalence_test()