# https://towardsdatascience.com/deep-learning-for-nlp-with-pytorch-and-torchtext-4f92d69052f

# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

# spacy vocab
# 先将补充之后的vocab和embeding保存为pkl

# stem 
# https://stackoverflow.com/questions/38763007/how-to-use-spacy-lemmatizer-to-get-a-word-into-basic-form


# torchtext Glove
# https://github.com/vincentzlt/torchtext/blob/master/torchtext/vocab.py

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv, pickle, re
import os, glob
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch, json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torch import nn
from torch.nn import CrossEntropyLoss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_NEIGHBOR = 10
def find_latest_folder(directory):
    return max(glob.glob(os.path.join(directory, '*.pt')), key=os.path.getmtime)

def _save_ckpt(args, model, output_dir = None, name=None):
    # output_dir = output_dir if output_dir is not None else "./ckpt/"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", output_dir)
    # model.module.save_pretrained(output_dir)
    torch.save(model.module.state_dict(), os.path.join(output_dir, name))

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, word, neighbor_s, neighbor_w, num_neighbor):
        """Constructs a InputExample.

        Args:

        """
        self.word = word
        self.neighbor_weight = neighbor_w
        self.neighbor_s = neighbor_s
        self.num_neighbor = num_neighbor

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, word_id, neighbor_id, neighbor_weight, alpha, num_neighbor, mask):
        self.word_id = word_id
        self.neighbor_id = neighbor_id
        self.neighbor_weight = neighbor_weight
        self.alpha = alpha
        self.num_neighbor = num_neighbor
        self.mask = mask

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_pickle(cls, input_file, _type = "train"):
        with open(input_file, "rb") as f:
            data = pickle.load(f)

        # return data
        # examples = {"vocab": data["vocab"], "embeddings": data["vectors"]}
        lines = []
        print("vocab size", len(data["conceptnet"]))
        for k,v in tqdm(data["conceptnet"].items()):
            # print(k)
            # word = k
            tmp_s = []
            tmp_w = []
            # print(k)
            # print(v)
            for nei in v:
                # print(nei)
                tmp_s.append(nei[0])
                # if nei[1] >= 1:
                #     tmp_w.append(0.2)
                # else:
                tmp_w.append(nei[1])
            tmp={}
            tmp["word"] = k
            tmp["neighbor"] = tmp_s
            tmp["neighbor_weight"] = tmp_w
            lines.append(tmp)
        # examples["lines"] = lines
        return lines
        
        
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.pkl")))
        return self._create_examples(
            self._read_pickle(os.path.join(data_dir, "train.pkl")), "train")
            # [:9400000]
            # [:9400000]
            # [:400000]

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pickle(os.path.join(data_dir, "dev.pkl")), "dev")[:100]

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pickle(os.path.join(data_dir, "test.pkl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, _type = "train"):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):

            word = line["word"]
            neighbor_s = line["neighbor"]
            neighbor_w = line["neighbor_weight"]
            neighbor_w.sort()
            neighbor_w = neighbor_w[::-1]
            examples.append(
                InputExample(word=word, neighbor_s=neighbor_s, neighbor_w=neighbor_w, num_neighbor = len(neighbor_w)))
        return examples

def convert_examples_to_features(examples, data):
    """Loads a data file into a list of `InputBatch`s."""
    # with open(input_file, "rb") as f:
    #     data = pickle.load(f)
    vocab = data["vocab"]
    stoi = {w:i for i,w in enumerate(vocab)}
    features = []

    for (ex_index, example) in tqdm(enumerate(examples)):
        # if ex_index > 500:
        #     break
        tmp = {}
        tmp["alpha"] = 20.0
        if example.word in stoi:
            tmp["stoi"] = stoi[example.word]
            # 第一阶段只训练GLOVE
            if tmp["stoi"] >= 400000:
                tmp["alpha"] = 0.0
                continue
        else:
            tmp["stoi"] = len(stoi) - 2
        
        neig = [len(stoi) - 1] * MAX_NEIGHBOR
        for wi,neighbor_w in enumerate(example.neighbor_s):
            if wi >= MAX_NEIGHBOR:
                break
            # if neighbor_w in vocab:
            #     neig.append(stoi[neighbor_w])
            # else:
            #     neig.append(len(stoi) - 2)
            if neighbor_w in stoi:
                neig[wi] = stoi[neighbor_w]
            else:
                neig[wi] = len(stoi) - 2
        # print(neig)
        tmp["neighbor_id"] = neig
        
        tmp["neighbor_weight"] = [0] * MAX_NEIGHBOR
        tmp["num_neighbor"] = example.num_neighbor
        if len(example.neighbor_weight) > MAX_NEIGHBOR:
            tmp["num_neighbor"] = MAX_NEIGHBOR
            example.neighbor_weight = example.neighbor_weight[:MAX_NEIGHBOR]
        # print(example.neighbor_weight)
        tmp["neighbor_weight"][:len(example.neighbor_weight)] = example.neighbor_weight
        
        mask = [0] * MAX_NEIGHBOR
        mask[:len(example.neighbor_weight)] = [1] * len(example.neighbor_weight)
        features.append(
            InputFeatures(word_id = tmp["stoi"], neighbor_id = tmp["neighbor_id"], neighbor_weight = tmp["neighbor_weight"], alpha = tmp["alpha"], num_neighbor = tmp["num_neighbor"], mask  = mask)
        )
    return features
        
class MyModel(nn.Module):
    def __init__(self, model_param, embedding):
        super().__init__()
        # with open("")

        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)

        # self.embedding_original = nn.Embedding.from_pretrained(embedding, freeze=True)
        
    # def forward(self, x, x_neighbor, weight):
    #     dif1 = self.embedding(x) - self.embedding_original(x)
    #     dif1_ = torch.sum(torch.square(dif1), axis=-1)
    #     # print(x)
    #     print(x.size(), x_neighbor.size())
    #     dif2  = self.embedding(x).unsqueeze(1) - self.embedding(x_neighbor)
    #     # print(dif2)
    #     dif2_ = torch.sum(torch.square(dif2), axis=-1)
    #     print(dif1_.size(), dif2_.size(), weight.size())
    #     alpha = 1.0
    #     loss = alpha * dif1_ + torch.sum(weight * dif2_, axis=-1)
    #     # print(loss.size())
    #     # return torch.mean(loss.unsqueeze(0), -1, keepdim=True)
    #     return loss.mean()

    def forward(self, x, x_original, x_neighbor, weight, alpha, num_neighbor, mask):
        # print(x.size(), x_original.size())
        dif1 = self.embedding(x) - x_original
        dif1_ = torch.sum(torch.square(dif1), axis=-1)
        # print(x)
        # print(x.size(), x_neighbor.size())
        dif2  = self.embedding(x).unsqueeze(1) - self.embedding(x_neighbor)
        print(mask)
        dif2 = dif2 * mask.unsqueeze(-1)

        # print(dif2)
        dif2_ = torch.sum(torch.square(dif2), dim=-1)
        # print(dif1_.size(), dif2_.size(), weight.size())
        # alpha = 100.0
        beta = 1.0
        # (b,l)
        # weight = torch.nn.functional.softmax(weight, dim = -1)
        
        dif2_ = torch.sum(weight * dif2_, dim=-1)
        # print("-------")
        # print("weight", weight)
        # print("dif1_",dif1_)
        print(dif1_, dif2_)
        loss = alpha * dif1_ + beta * dif2_
        # print(loss.size())
        # return torch.mean(loss.unsqueeze(0), -1, keepdim=True)
        return loss.mean()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--sleep",
                        default=0,
                        type=int,
                        required=True,
                        help="sleep time.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--feature_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_labels",
                        default=2,
                        type=int,
                        help="class labels")
    parser.add_argument("--name_save_steps",
                        default=50,
                        type=int,
                        help="save model steps")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="cache dir")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--use_wklm",
                        default=False,
                        action='store_true',
                        help="Whether to use wklm")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--lr_scheduler_type",
                        default="linear",
                        type=str,
                        required=False,
                        help="cache dir")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--use_ft",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--use_fast_tokenizer",
                        default=False,
                        action='store_true',
                        help="Whether not to use fast tokenizer")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--max_path_num', 
                        type=int, 
                        default=5,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    processor = MrpcProcessor()
    label_list = processor.get_labels()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
                            
    train_examples = None
    num_train_steps = 1000
    if args.do_train:
        print("Count number")
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    with open("data-lm4kg/train.pkl", "rb") as f:
        d = pickle.load(f)
        # print(d["vectors"][:2])1
    model = MyModel(args, d["vectors"])

    if len(args.model_name_or_path.strip()) > 5:
        print("LOAD CKPT", args.model_name_or_path)
        ckpt = torch.load(args.model_name_or_path)
        model.load_state_dict(ckpt,strict=False)

    if args.fp16:
        model.half()
    print(device)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        # {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    
    from transformers import AdamW
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr = args.learning_rate)

    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
                args.lr_scheduler_type,
                optimizer,
                num_warmup_steps = 0,
                num_training_steps = t_total,
            )
    # loss_fct = CrossEntropyLoss()

    best_model = {"eval_result":100000000, "model_path":None}
    global_step = 0
    
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    with torch.no_grad():
        embedding_original = nn.Embedding.from_pretrained(d["vectors"], freeze=True).cpu()

    if args.do_train:
        # print(len(train_examples))
        if not os.path.exists(os.path.join(args.feature_dir,"train.pkl")):
            train_features = convert_examples_to_features(
                train_examples, d)
            with open(os.path.join(args.feature_dir,"train.pkl"),"wb") as f:
                pickle.dump(train_features, f)
        else:
            with open(os.path.join(args.feature_dir,"train.pkl"),"rb") as f:
                train_features = pickle.load(f)
        
        # print(len(train_features))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_word_ids = torch.tensor([f.word_id for f in train_features], dtype=torch.long)
        all_neighbor_ids = torch.tensor([f.neighbor_id for f in train_features], dtype=torch.long)
        all_neighbor_weights = torch.tensor([f.neighbor_weight for f in train_features], dtype=torch.float32)
        all_alpha = torch.tensor([f.alpha for f in train_features], dtype=torch.float32)
        all_num_neighbor = torch.tensor([f.num_neighbor for f in train_features], dtype=torch.float32)
        all_neighbor_mask = torch.tensor([f.mask for f in train_features], dtype=torch.float32)
        train_data = TensorDataset(all_word_ids, all_neighbor_ids, all_neighbor_weights, all_alpha, all_num_neighbor, all_neighbor_mask)
        # print(all_segment_ids.size(), all_input_ids.size())
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        loss = None
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            # model.train() # in case evaluation
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                word_ids, neighbor_ids, neighbor_weights, alpha, num_neighbor, mask = batch
                # print(word_ids.size())
                x_original = embedding_original(word_ids.cpu()).detach()
                loss = model(word_ids, x_original, neighbor_ids, neighbor_weights, alpha, num_neighbor, mask)
                # loss = output.loss

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                print("LOSS", loss)
                loss.backward()
                # print(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += word_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    lr_scheduler.step()
                    global_step += 1

                    if global_step % args.name_save_steps == 0 and global_step > 1:
                        _save_ckpt(args, model, output_dir = args.output_dir, name = "{}.pt".format(global_step))

                        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                            model_path = find_latest_folder(args.output_dir)
                            
                            
                            logger.info("****** Only Eval 5000 samples ******")
                            eval_examples = processor.get_dev_examples(args.data_dir)
                            if not os.path.exists(os.path.join(args.feature_dir,"dev.pkl")):
                                eval_features = convert_examples_to_features(
                                    eval_examples, d)
                                eval_features = eval_features[:10]
                                with open(os.path.join(args.feature_dir,"dev.pkl"),"wb") as f:
                                    pickle.dump(eval_features, f)
                            else:
                                with open(os.path.join(args.feature_dir,"dev.pkl"),"rb") as f:
                                    eval_features = pickle.load(f)
                                    eval_features = eval_features[:10]
                            logger.info("***** Running evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)

                            eval_word_ids = torch.tensor([f.word_id for f in eval_features], dtype=torch.long)
                            eval_neighbor_ids = torch.tensor([f.neighbor_id for f in eval_features], dtype=torch.long)
                            eval_neighbor_weights = torch.tensor([f.neighbor_weight for f in eval_features], dtype=torch.float32)
                            eval_alpha = torch.tensor([f.alpha for f in eval_features], dtype=torch.float32)
                            eval_num_neighbor = torch.tensor([f.num_neighbor for f in eval_features], dtype=torch.float32)
                            eval_neighbor_mask = torch.tensor([f.mask for f in eval_features], dtype=torch.float32)
                            eval_data = TensorDataset(eval_word_ids, eval_neighbor_ids, eval_neighbor_weights, eval_alpha, eval_num_neighbor, eval_neighbor_mask)
                            # Run prediction for full data
                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                            model.eval()
                            eval_loss, eval_accuracy = 0, 0
                            nb_eval_steps, nb_eval_examples = 0, 0
                            output = None
                            for word_ids, neighbor_ids, neighbor_weights, alpha, num_neighbor, neighbor_mask in eval_dataloader:
                                word_ids = word_ids.to(device)
                                neighbor_ids = neighbor_ids.to(device)
                                neighbor_weights = neighbor_weights.to(device)
                                alpha = alpha.to(device)
                                num_neighbor = num_neighbor.to(device)
                                neighbor_mask = neighbor_mask.to(device)

                                with torch.no_grad():
                                    x_original = embedding_original(word_ids.cpu()).detach()
                                    loss = model(word_ids, x_original, neighbor_ids, neighbor_weights, alpha, num_neighbor, neighbor_mask)
                                    # loss = output.loss
                                    
                                    tmp_eval_loss = loss
                                    logits = loss

                                logits = logits.detach().cpu().numpy()
                                # label_ids = label_ids.to('cpu').numpy()
                                # tmp_eval_accuracy = accuracy(logits, label_ids)

                                eval_loss += tmp_eval_loss.mean().item()
                                # eval_accuracy += tmp_eval_accuracy
                                # sys.exit()
                                # print("name exit----")
                                nb_eval_examples += word_ids.size(0)
                                nb_eval_steps += 1

                            eval_loss = eval_loss / nb_eval_steps
                            # eval_accuracy = eval_accuracy / nb_eval_examples
                            
                            model.train() # in case evaluation
                            if eval_loss < best_model["eval_result"]:
                                best_model["eval_result"] = eval_loss
                                best_model["model_path"] = model_path

                            result = {'eval_loss': eval_loss,
                                    # 'eval_accuracy': eval_accuracy,
                                    'global_step': global_step,
                                    'loss': tr_loss/nb_tr_steps}

                            result.update(best_model)

                            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                            with open(output_eval_file, "w") as writer:
                                logger.info("***** Eval results *****")
                                for key in sorted(result.keys()):
                                    logger.info("  %s = %s", key, str(result[key]))
                                    writer.write("%s = %s\n" % (key, str(result[key])))
    
    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model_path = best_model["model_path"]
        if model_path is None:
            model_path = args.model_name_or_path
        model = MyModel(args, d["vectors"])

        d = torch.load(model_path)
        model.load_state_dict(d,strict=False)

        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        test_examples = processor.get_test_examples(args.data_dir)
        if not os.path.exists(os.path.join(args.feature_dir,"test.pkl")):
            test_features = convert_examples_to_features(
                test_examples, d)
            with open(os.path.join(args.feature_dir,"test.pkl"),"wb") as f:
                pickle.dump(test_features, f)
        else:
            with open(os.path.join(args.feature_dir, "test.pkl"),"rb") as f:
                test_features = pickle.load(f)
        logger.info("***** Running Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_word_ids = torch.tensor([f.word_id for f in test_features], dtype=torch.long)
        test_neighbor_ids = torch.tensor([f.neighbor_id for f in test_features], dtype=torch.long)
        test_neighbor_weights = torch.tensor([f.neighbor_weight for f in test_features], dtype=torch.float32)
        test_alpha = torch.tensor([f.alpha for f in test_features], dtype=torch.float32)
        test_num_neighbor = torch.tensor([f.num_neighbor for f in test_features], dtype=torch.float32)
        test_neighbor_mask = torch.tensor([f.mask for f in test_features], dtype=torch.float32)
        test_data = TensorDataset(test_word_ids, test_neighbor_ids, test_neighbor_weights, test_alpha, test_num_neighbor, test_neighbor_mask)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        output = None
        predict_scores = []
        for word_ids, neighbor_ids, neighbor_weights, alpha, num_neighbor, neighbor_mask in test_dataloader:
            word_ids = word_ids.to(device)
            neighbor_ids = neighbor_ids.to(device)
            neighbor_weights = neighbor_weights.to(device)
            alpha = alpha.to(device)
            num_neighbor = num_neighbor.to(device)
            neighbor_mask = neighbor_mask.to(device)

            with torch.no_grad():
                x_original = embedding_original(neighbor_ids.cpu()).detach()
                loss = model(word_ids, x_original, neighbor_ids, neighbor_weights, alpha, num_neighbor, neighbor_mask)
                # loss = output.loss
                
                tmp_eval_loss = loss
                logits = loss

            logits = logits.detach().cpu().numpy()
            # label_ids = label_ids.to('cpu').numpy()
            # tmp_eval_accuracy = accuracy(logits, label_ids)
            predict_scores.append([logits, word_ids.cpu().numpy()])

            eval_loss += tmp_eval_loss.mean().item()
            # eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += word_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        # eval_accuracy = eval_accuracy / nb_eval_examples

        # if eval_accuracy > best_model["eval_result"]:
        best_model["test_result"] = eval_loss
        best_model["model_path"] = model_path

        result = {'test_loss': eval_loss,
                'global_step': global_step}
                
        result.update(best_model)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        np.save(os.path.join(args.output_dir, "predict_scores.npy"), predict_scores)
            

if __name__ == "__main__":
    main()
