import os
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from experiment.bert.run_classifier import MrpcProcessor, logger, convert_examples_to_features, \
    set_optimizer_params_grad, copy_optimizer_params_to_model, accuracy, p_r_f1, tp_pcount_gcount, \
    InputExample

bert_model = "bert-base-uncased"

# Data Directory
FILE_PATH = {
    # Stance
    "stance_data": "../data/dataset/perspective_stances/",
    "stance_model_dir": "../model/stance",

    # Relevance
    "relevance_data": "../data/dataset/perspective_relevance/",
    "relevance_model_dir": "../model/relevance",

    # Evidence
    "evidence_data": "../data/dataset/perspective_evidence/",
    "evidence_model_dir": "../model/evidence",

    # Equivalence
    "equivalence_data": "../data/dataset/perspective_equivalence/",
    "equivalence_model_dir": "../model/equivalence",
}

# Default config file
DEFAULT_CONFIG = {
    "bert_model": "bert-base-uncased",
    "max_seq_length": 128,
    "do_lower_case": False,
    "train_batch_size": [16, 32],
    "eval_batch_size": 8,
    "learning_rate": [3e-5, 2e-5, 5e-5],
    "num_train_epochs": 5,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128,
    "task_name": "",
}

"""
TODO: Improvements
1. Look at where they cached the bert model, if it's under /home/ or /tmp, move it somewhere else
"""


class BertBaseline:
    def __init__(self, saved_model=None, **kwargs):
        """
        :param saved_model: path to trained model
        :param kwargs: see DEFAULT_CONFIG above
        """
        self._config = {**DEFAULT_CONFIG, **kwargs}

        print(self._config)

        if self._config["local_rank"] == -1 or self._config["no_cuda"]:
            print("here")
            self._device = torch.device("cuda" if torch.cuda.is_available() and not self._config["no_cuda"] else "cpu")
            print(self._device)
            self._n_gpu = torch.cuda.device_count()
        else:
            self._device = torch.device("cuda", self._config["local_rank"])
            self._n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
            if self._config["fp16"]:
                logger.info("16-bits training currently not supported in distributed training")
                self._config["fp16"] = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device %s n_gpu %d distributed training %r", self._device, self._n_gpu,
                    bool(self._config["local_rank"] != -1))

        if self._config["gradient_accumulation_steps"] < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self._config["gradient_accumulation_steps"]))

        random.seed(self._config["seed"])
        np.random.seed(self._config["seed"])
        torch.manual_seed(self._config["seed"])
        if self._n_gpu > 0:
            torch.cuda.manual_seed_all(self._config["seed"])

        self._processor = MrpcProcessor()

        self._tokenizer = BertTokenizer.from_pretrained(self._config["bert_model"],
                                                        do_lower_case=self._config["do_lower_case"])

        self._init_model(saved_model)

    def _init_model(self, saved_model=None):
        # Load pre-trained BERT
        if saved_model:
            print("Loading the pre-trained model from: " + saved_model)
            # if loading on a cpu:
            model_state_dict = torch.load(saved_model, map_location='cpu')
            # model_state_dict = torch.load(saved_model)
            self._model = BertForSequenceClassification.from_pretrained(bert_model, state_dict=model_state_dict, num_labels=2)
        else:
            cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(self._config["local_rank"]))
            self._model = BertForSequenceClassification.from_pretrained(self._config["bert_model"],
                                                                        cache_dir=cache_dir,
                                                                        num_labels=len(self._processor.get_labels()))

        if self._config["fp16"]:
            self._model.half()
        self._model.to(self._device)
        if self._config["local_rank"] != -1:
            model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self._config["local_rank"]],
                                                              output_device=self._config["local_rank"])
        elif self._n_gpu > 1:
            _model = torch.nn.DataParallel(self._model)

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

        # if os.path.exists(output_dir) and os.listdir(output_dir):
        #     raise ValueError("Output directory ({}) already exists and is not emp1ty.".format(output_dir))

        # Log validation results to a log file in the output directory
        __log_path = os.path.join(output_dir, "{}_train.log".format(self._config["task_name"]))

        # Prepare training examples
        train_examples = self._processor.get_train_examples(train_data_dir)
        dev_examples = self._processor.get_dev_examples(train_data_dir)

        label_list = self._processor.get_labels()
        tokenizer = self._tokenizer
        train_features = convert_examples_to_features(train_examples, label_list,
                                                      self._config["max_seq_length"], tokenizer)
        dev_features = convert_examples_to_features(dev_examples, label_list,
                                                    self._config["max_seq_length"], tokenizer)

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

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  --- Sweeping Parameters ---")
        logger.info("\tLearning Rate --> {}".format(self._config["learning_rate"]))
        logger.info("\tBatch Size --> {}".format(self._config["train_batch_size"]))

        # Do parameter sweep for training
        for __lr in self._config["learning_rate"]:
            for __bs in self._config["train_batch_size"]:

                # Save model from each parameter into separate directory
                __directory_name = "lr{}_bs{}".format(__lr, __bs)
                __current_output_dir = os.path.join(output_dir, __directory_name)

                if not os.path.exists(__current_output_dir):
                    os.makedirs(__current_output_dir, exist_ok=True)

                __train_batch_size = int(__bs / self._config["gradient_accumulation_steps"])

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
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                     'weight_decay_rate': 0.0}
                ]

                num_train_steps = int(
                    len(train_examples) / __train_batch_size / self._config["gradient_accumulation_steps"] *
                    self._config["num_train_epochs"])

                t_total = num_train_steps
                if self._config["local_rank"] != -1:
                    t_total = t_total // torch.distributed.get_world_size()

                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=__lr,
                                     warmup=self._config["warmup_proportion"],
                                     t_total=t_total)

                logger.info("  --- Current Parameters ---")
                logger.info("\tLearning Rate = {}".format(__lr))
                logger.info("\tBatch Size = {}".format(__bs))
                logger.info("\tNum steps = {}".format(num_train_steps))

                with open(__log_path, 'a+') as fout:
                    fout.write("--- Current Parameters ---\n")
                    fout.write("Learning Rate = {}\n".format(__lr))
                    fout.write("Batch Size = {}\n".format(__bs))
                    fout.write("Num steps = {}\n".format(num_train_steps))
                    fout.write("\n")

                # Begin Training
                if self._config["local_rank"] == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)

                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=__bs)

                self._model.train()
                global_step = 0
                loss_scale = self._config["loss_scale"]
                for _epoch in trange(int(self._config["num_train_epochs"]), desc="Epoch"):
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0
                    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                        batch = tuple(t.to(self._device) for t in batch)
                        input_ids, input_mask, segment_ids, label_ids = batch
                        loss = self._model(input_ids, segment_ids, input_mask, label_ids)
                        if self._n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if self._config["fp16"] and loss_scale != 1.0:
                            # rescale loss for fp16 training
                            # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                            loss = loss * loss_scale
                        if self._config["gradient_accumulation_steps"] > 1:
                            loss = loss / self._config["gradient_accumulation_steps"]
                        loss.backward()
                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1
                        if (step + 1) % self._config["gradient_accumulation_steps"] == 0:
                            if self._config["fp16"] or self._config["optimize_on_cpu"]:
                                if self._config["fp16"] and loss_scale != 1.0:
                                    # scale down gradients for fp16 training
                                    for param in self._model.parameters():
                                        if param.grad is not None:
                                            param.grad.data = param.grad.data / loss_scale
                                is_nan = set_optimizer_params_grad(param_optimizer, self._model.named_parameters(),
                                                                   test_nan=True)
                                if is_nan:
                                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                    loss_scale = loss_scale / 2
                                    self._model.zero_grad()
                                    continue
                                optimizer.step()
                                copy_optimizer_params_to_model(self._model.named_parameters(), param_optimizer)
                            else:
                                optimizer.step()
                            self._model.zero_grad()

                            global_step += 1

                    # At the end of each epoch, validate on dev
                    p, r, f1 = self._evaluate(dev_data)

                    with open(__log_path, 'a+') as fout:
                        fout.write("Epoch #{}\n".format(_epoch))
                        fout.write("\tMicro Precision = {}\n".format(p))
                        fout.write("\tMicro Precision = {}\n".format(p))
                        fout.write("\tMicro Recall = {}\n".format(r))
                        fout.write("\tMicro F1 = {}\n".format(f1))
                        fout.write("\n")

                    # Save model after every epoch
                    __save_name = "{}_epoch-{}.pth".format(self._config["task_name"], _epoch)
                    torch.save(self._model.state_dict(), os.path.join(__current_output_dir, __save_name))

                # Reinitialize model weights
                self._init_model()

    def evaluate(self, data_dir, save_score_path=None):
        """
        Directory that contains a "test.tsv"
        :param data_dir:
        :return:
        """
        test_examples = self._processor.get_test_examples(data_dir)
        label_list = self._processor.get_labels()
        tokenizer = self._tokenizer

        test_features = convert_examples_to_features(test_examples, label_list,
                                                     self._config["max_seq_length"], tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return self._evaluate(test_data, save_score_path)


    def _evaluate(self, eval_data, save_score_path=None):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", self._config["eval_batch_size"])

        score_fout = None
        if save_score_path:
            score_fout = open(save_score_path, 'w')

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self._config["eval_batch_size"])

        self._model.eval()

        eval_tp, eval_pred_c, eval_gold_c = 0, 0, 0
        eval_loss, eval_macro_p, eval_macro_r = 0, 0, 0

        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(self._device)
            input_mask = input_mask.to(self._device)
            segment_ids = segment_ids.to(self._device)
            label_ids = label_ids.to(self._device)

            with torch.no_grad():
                tmp_eval_loss = self._model(input_ids, segment_ids, input_mask, label_ids)
                logits = self._model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            if score_fout:
                batch_logits_str = "\n".join(",".join('%0.3f' %x for x in y) for y in logits)
                score_fout.write(batch_logits_str)
                score_fout.write("\n")

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

        logger.info("  Micro Precision = {}".format(eval_micro_p))
        logger.info("  Micro Recall = {}".format(eval_micro_r))
        logger.info("  Micro F1 = {}".format(eval_micro_f1))

        if score_fout:
            score_fout.close()

        return eval_micro_p, eval_micro_r, eval_micro_f1

    def predict(self, sent1, sent2):
        """
        Predict on a single sentence pair (sent1, sent2); the choice of the inputs depends on the task at hand:
         - Relevance (claim, perspective)
         - Stance (claim, perspective)
         - Equivalence (claim + perspective1, perspective2)
         - Evidence (claim + perspective, evidence)
        :param example:
        :param sent1
        :param sent2
        :return the confidence value of the output label
        """
        label_list = self._processor.get_labels()
        example = InputExample(guid="dummy", text_a=sent1, text_b=sent2, label=label_list[0])
        feature = convert_examples_to_features([example], label_list,
                                               self._config["max_seq_length"], self._tokenizer)[0]

        self._model.eval()

        with torch.no_grad():
            input_ids_tensor = torch.tensor([feature.input_ids])
            segment_ids_tensor = torch.tensor([feature.segment_ids])
            input_mask_tensor = torch.tensor([feature.input_mask])

            output = self._model(input_ids_tensor, segment_ids_tensor, input_mask_tensor)
            # print(output)
            return output.detach().cpu().numpy()[0]


def stance_train():
    data_dir = FILE_PATH['stance_data']
    model_dir = FILE_PATH['stance_model_dir']
    bb = BertBaseline(task_name="perspectrum_stance", saved_model=None)
    bb.train(data_dir, model_dir)


def stance_evaluation(model_path):
    data_dir = FILE_PATH['stance_data']

    bb = BertBaseline(task_name="perspectrum_stance", saved_model=model_path)
    bb.evaluate(data_dir)


def equivalence_train():
    data_dir = FILE_PATH['equivalence_data']
    model_dir = FILE_PATH['equivalence_model_dir']

    bb = BertBaseline(task_name="perspectrum_equivalence", saved_model=None)
    bb.train(data_dir, model_dir)


def equivalence_evaluation(model_path):
    data_dir = FILE_PATH['equivalence_data']

    bb = BertBaseline(task_name="perspectrum_equivalence", saved_model=model_path)
    bb.evaluate(data_dir)


def relevance_train():
    data_dir = FILE_PATH['relevance_data']
    model_dir = FILE_PATH['relevance_model_dir']

    bb = BertBaseline(task_name="perspectrum_relevance", saved_model=None)
    bb.train(data_dir, model_dir)


def relevance_evaluation(model_path):
    data_dir = FILE_PATH['relevance_data']

    bb = BertBaseline(task_name="perspectrum_relevance", saved_model=model_path)
    bb.evaluate(data_dir)


def evidence_train():
    # Since evidence is usually long, we need to lift the max sequence length and lower the batch size.
    _config = {
        "bert_model": bert_model,
        "max_seq_length": 512,
        "do_lower_case": False,
        "train_batch_size": [4, 8],
        "eval_batch_size": 8,
        "learning_rate": [3e-5, 2e-5],
        "num_train_epochs": 5,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 128,
        "task_name": "perspectrum_evidence",
    }

    data_dir = FILE_PATH['evidence_data']
    model_dir = FILE_PATH['evidence_model_dir']

    bb = BertBaseline(None, **_config)
    bb.train(data_dir, model_dir)


def test_models():
    bb = BertBaseline(task_name="perspectrum_relevane",
                      saved_model="/Users/daniel/ideaProjects/perspective/model/relevance/perspectrum_relevance_lr2e-05_bs32_epoch-0.pth",
                      no_cuda=True)
    print(bb.predict("123", "345"))


if __name__ == "__main__":
    test_models()
    # stance_train()
    stance_evaluation("/scratch/sihaoc/project/perspective/model/stance/lr2e-05_bs16/perspectrum_stance_epoch-0.pth")
    # equivalence_train()
    # equivalence_evaluation("/scratch/sihaoc/project/perspective/model/equivalence/lr2e-05_bs16/perspectrum_equivalence_epoch-0.pth")
    # relevance_train()
    # evidence_train()

