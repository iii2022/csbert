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

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_PATH=5

def find_latest_folder(directory):
    return max(glob.glob(os.path.join(directory, '*/')), key=os.path.getmtime)

def _save_ckpt(args, model, tokenizer, output_dir = None):
    # output_dir = output_dir if output_dir is not None else "./ckpt/"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", output_dir)
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # if not isinstance(model, PreTrainedModel):
    #     logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
    #     state_dict = model.state_dict()
    #     torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    # else:
    # state_dict = model.state_dict()
    # torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    model.module.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))

def model_field(field, input_ids, label_ids, model, sep_ids, input_mask, segment_ids):
    # slid mask
    fill_ = (sep_ids == field)
    # tmp_mask = input_mask + fill_ * -1e12
    tmp_mask = input_mask * fill_
    tmp_mask[:,0] = input_mask[:,0]
    # print(field, tmp_mask)
    
    have_ = (sep_ids == field)
    tmp_input = input_ids * have_
     
    tmp_input[:,0] = input_ids[:,0]
    # print(field, tmp_input)

    tmp_segment_ids = torch.zeros_like(segment_ids, dtype = torch.long)
    # model.eval()
    output, CLS = model(tmp_input, attention_mask = tmp_mask, position_ids = tmp_segment_ids, labels = None)
    # model.train()
    return CLS

# class FTClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 2 or 3 classes
#         self.dense = nn.Linear(768, 2)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, hidden_states_dict, sep_ids):
        
#         length = torch.sum(sep_ids==0,dim = -1)+1e-10
#         triple_t = torch.sum(hidden_states_dict["triple"],dim=1, keep_dims=True)

#         length = torch.sum(sep_ids==1,dim = -1)+1e-10
#         start_t = torch.sum(hidden_states_dict["start_synset"],dim=1, keep_dims=True)
        
#         length = torch.sum(sep_ids==2,dim = -1)+1e-10
#         end_t = torch.sum(hidden_states_dict["end_synset"],dim=1, keep_dims=True)
#         c = 2
#         paths = []
#         for tensor in hidden_states_dict["path"]:
#             c += 1
#             length = torch.sum(sep_ids==c,dim = -1)+1e-10
#             path = torch.sum(tensor, dim = 1, keep_dims=True)
#             paths.append(path)
#         hidden_states = torch.concat([triple_t, start_t, end_t] + paths, dim=1)
#         hidden_states = torch.sum(hidden_states, dim=1)
#         hidden_states = self.dense(hidden_states)
        
#         return hidden_states

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, conceptnet_triple, context=None, start_synset=None, end_synset=None, hownet_paths=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.context = context
        self.conceptnet_triple = conceptnet_triple
        self.start_synset = start_synset
        self.end_synset = end_synset
        self.hownet_paths = hownet_paths
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sep_ids, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.sep_ids = sep_ids


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

        def process_node(s):
            if s is None:
                return ""
            s = s.strip()
            # s = re.sub(r"_","",s)
            arr = s.split("|")
            if len(arr) > 1:
                if arr[0][0]=="{":
                    return arr[0][1:]
            else:
                if s[0] == "{" and s[-1] == "}":
                    s = s[1:-1]
                return s
            # 如果不属于以上情况怎么办，返回""否则就给你返回一None导致报错
            return ""

        def process_edge(s):
            s = s.strip()
            # s_ = re.sub(r"\s","_",s)
            s_ = re.sub(r"_"," ",s)
            return s_
        """
        # def parse_paths(paths):
        #     # sort paths by length
        #     paths.sort(key=lambda s: len(s))
        #     paths = paths[:1]
        #     res = []
        #     for _,path in enumerate(paths):
        #         tmp = []
        #         for i,e in enumerate(path):
        #             if i == 0:
        #                 # tmp.append("$")
        #                 tmp.append(process_node(e[0]))
        #             # tmp.append("@")
        #             tmp.append(process_edge(e[1]))
        #             # tmp.append("$")
        #             tmp.append(process_node(e[2]))
        #         # tmp = tmp[:-1]
        #         res.append(tmp)

        #     # 在这里可以对path按照路径长度排序，随机选取
        #     # 只要注释掉返回值就实现了ablation
            # return res
        """

        def parse_paths(paths, start_root, end_root):
            # sort paths by length
            paths.sort(key=lambda s: len(s))
            # 只使用一条路径啊！！！！
            # paths = paths[:1]

            res = []
            for _,path in enumerate(paths):
                if True:
                # if start_root in path[0][0] and end_root == path[-1][-1]:
                    tmp = []
                    for i,e in enumerate(path):
                        if i == 0:
                            # tmp.append("$")
                            tmp.append(process_node(e[0]))
                        # tmp.append("@")
                        tmp.append(process_edge(e[1]))
                        # tmp.append("$")
                        tmp.append(process_node(e[2]))
                    # tmp = tmp[:-1]
                    if "".join(tmp).strip() == "":
                        continue
                    if len(tmp) > 0:
                        res.append(tmp)

            while len(res) < MAX_PATH:
                res += ["N"]
            # 在这里可以对path按照路径长度排序，随机选取
            # 只要注释掉返回值就实现了ablation
            
            return res[:MAX_PATH]

        def parse_triple(s):

            def process_uri(t):
                arr = t.split("/")
                if len(arr) > 1:
                    t_ = re.sub("_"," ",arr[3])
                    return t_
                else:
                    t_ = re.sub("_"," ",t)
                    return t_

            arr = s.strip().split("####")
            # print(arr)
            assert(len(arr) ==  3)
            sub = process_uri(arr[0])
            obj = process_uri(arr[2])
            pre = arr[1][3:]
            st = "{} {} {}".format(sub, pre, obj)
            return st

        # def parse_synset(s):
        #     if s.strip() == "":
        #         return
        #     s = re.sub(r"'","\"", s)
        #     synset = json.loads(s)
        #     # synset = s
        #     print(synset[0])
        #     seq = []
        #     for triple in synset:
        #         seq.append(triple[0])
        #         seq.append(triple[-1])

        #     seq_ = " ".join(list(sorted(set(seq))))
        #     return seq_

        def parse_synset(sysets):
            sysets = re.sub(r"'","\"", sysets)
            # print("synset", sysets)
            if sysets.strip() == "":
                return "", ""
            js = json.loads(sysets)
            res = []
            for i,x in enumerate(js):
                if x[0].lower() not in res and x[0] not in res:
                    res.append(process_node(x[0]))
                if x[2].lower() not in res and x[2] not in res:
                    res.append(process_node(x[2]))
            # return res
            # seq_ = " ".join(list(sorted(set(res))))
            seq_ = " ".join(res)
            # do not use synset tree
            return seq_, js[0][-1]
            # return "", js[0][-1]


        with open(input_file,"rb") as f:
            data = pickle.load(f)
        # return data
        lines = []
        for i, row in tqdm(enumerate(data)):
            # if i > 1000:
            #     break
            tmp = {}
            tmp["statement_id"] = row["statement_id"]
            tmp["conceptnet_triple"]  = parse_triple(row["conceptnet_triple"])
            
            tmp["start_synset"], start_root = parse_synset(row["start_synset"])
            tmp["end_synset"], end_root = parse_synset(row["end_synset"])

            tmp["hownet_paths"] = parse_paths(row["hownet_paths"], start_root, end_root)
            if row["label"] == "":
                tmp["label"] = 0
            else:
                tmp["label"] = 1 if row["label"] > 0.1 else 0
            tmp["context"] = ""
            
            lines.append(tmp)
        return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.pkl")))
        return self._create_examples(
            self._read_pickle(os.path.join(data_dir, "train.pkl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pickle(os.path.join(data_dir, "dev.pkl")), "dev")

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
            examples.append(
                InputExample(guid=line["statement_id"], conceptnet_triple=line["conceptnet_triple"], context=line["context"], start_synset=line["start_synset"], end_synset=line["end_synset"], hownet_paths=line["hownet_paths"], label=line["label"]))
        return examples

# def process_sep(sep):
    

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    """
    input: {"label":0.5, "statement_id":"node111", "context":"...", "conceptnet_triple":"...","hownet_paths":"..."}
    output:
    [CLS]cat,usedfor,paly[SEP0]You can use a cat to play with and make you laugh
    [SEP1]{livestock|牲畜},MaterialOf,{edible|食物},subClassOf,{artifact|人工物},RelatedTo,{implement|器具},subclassof-1,{musictool|乐器}
    [SEP2]...
    """

    features = []
    sep = {}
    for (ex_index, example) in enumerate(examples):
        tokens = []
        segment_ids = []
        sep_ids = []
        if example.conceptnet_triple.strip() != "":
            tokens_triple = tokenizer.tokenize(example.conceptnet_triple)
            tokens.append("[CLS]")
            tokens += tokens_triple
            # tokens.append("[SEP]")
            sep_ids += (len(tokens) - len(segment_ids)) * [0]
            segment_ids += (len(tokens) - len(segment_ids)) * [0]
            

        # if example.context.strip() != "":
        #     tokens_context = tokenizer.tokenize(example.context)
        #     # tokens.append("[unused0]")
        #     tokens.append("[SEP]")
        #     tokens += tokens_context
        #     segment_ids += (len(tokens) - len(segment_ids)) * [1]

        # if example.start_synset != "" and example.end_synset != "":
        tokens.append("[unused0]")
        sep["start_synset"] = len(tokens)-1
        tokens_start_synset = tokenizer.tokenize(example.start_synset)
        tokens += tokens_start_synset
        sep_ids += (len(tokens) - len(segment_ids)) * [1]
        segment_ids += (len(tokens) - len(segment_ids)) * [0]
        
            
        # if example.start_synset != "" and example.end_synset != "":
        tokens.append("[unused1]")
        sep["end_synset"] = len(tokens)-1
        tokens_end_synset = tokenizer.tokenize(example.end_synset)
        tokens += tokens_end_synset
        sep_ids += (len(tokens) - len(segment_ids)) * [2]
        segment_ids += (len(tokens) - len(segment_ids)) * [0]
        

        pos = 3
        sep["path"] = []
        # hownet_paths = []
        if example.hownet_paths:
            for path_i, path in enumerate(example.hownet_paths):
                path_ = " ".join(path)
                if path_.strip() == "":
                    continue
                tokens_path = tokenizer.tokenize(path_)
                # hownet_paths.append(tokens_path)
                
                # not add incomplete path
                if len(tokens_path) + len(tokens) >= max_seq_length:
                    break
                # tokens.append("[unused{}]".format(path_i+1))
                tokens.append("[unused2]")
                sep["path"].append(len(tokens)-1)
                tokens += tokens_path
                
                sep_ids += (len(tokens) - len(segment_ids)) * [pos]
                segment_ids += (len(tokens) - len(segment_ids)) * [0]
                pos += 1
        
        tokens.append("[SEP]")
        sep_ids += (len(tokens) - len(segment_ids)) * [pos-1]
        segment_ids += (len(tokens) - len(segment_ids)) * [0]
        
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        sep_ids = sep_ids[:max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            # input_mask.append(-1e12)
            input_mask.append(0)
            segment_ids.append(0)
            sep_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(sep_ids) == max_seq_length

        # label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "sep_ids: %s" % " ".join([str(x) for x in sep_ids]))
            logger.info("label: (id = %d)" % (example.label))

        features.append(
            InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=example.label,
                    sep_ids = sep_ids,
                    guid = example.guid))
    # return features[:50000]
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
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
                        default=5e-5,
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

    processors = {
        "cola": None,
        "mnli": None,
        "hownet": MrpcProcessor,
    }

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

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # https://huggingface.co/transformers/main_classes/configuration.html
    # configuration_utils.py
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=args.cache_dir,
        # id2label=label_map,
        # label2id={label: i for i, label in enumerate(labels)},
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )
    # if model_args.model_name_or_path == "gpt2":
    #     ## tokenizer.pad_token = tokenizer.eos_token
    #     # special_tokens = {'cls_token': '<|cls|>','pad_token':tokenizer.eos_token,'sep_token':'<|sep|>'}
    #     # num_add_toks = tokenizer.add_special_tokens(special_tokens)
    #     # print("+++++++++++++++", num_add_toks, tokenizer.pad_token_id)
    #     tokenizer.add_special_tokens({'pad_token': "GD"})
    flag = False
    if tokenizer.pad_token is None:
        flag = True
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = None
    num_train_steps = 1000
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, 
    #             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    # ftclassifier = FTClassifier()
    if args.use_wklm:
        print("load wklm ***************", args.wklm_model_path)
        d = torch.load(args.wklm_model_path)
        for nk in ["generator_predictions.LayerNorm.weight", "generator_predictions.LayerNorm.bias", "generator_predictions.dense.weight", "generator_predictions.dense.bias", "generator_lm_head.weight", "generator_lm_head.bias"]:
            if nk in d:
                del d[nk]
           
        model.load_state_dict(d,strict=False)
    
    if args.fp16:
        model.half()
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
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    #     ]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=t_total)
    from transformers import AdamW
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr = args.learning_rate)

    from transformers import get_scheduler
    print("warmup is 0")
    lr_scheduler = get_scheduler(
                args.lr_scheduler_type,
                optimizer,
                num_warmup_steps = 0,
                num_training_steps=t_total,
            )
    # loss_fct = CrossEntropyLoss()

    best_model = {"eval_result":-100, "model_path":None}
    global_step = 0
    
    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir)

    if args.do_train:
        print(len(train_examples))
        if not os.path.exists(os.path.join(args.feature_dir,"train.pkl")):
            train_features = convert_examples_to_features(
                train_examples, args.max_seq_length, tokenizer)
            with open(os.path.join(args.feature_dir,"train.pkl"),"wb") as f:
                pickle.dump(train_features, f)
        else:
            with open(os.path.join(args.feature_dir,"train.pkl"),"rb") as f:
                train_features = pickle.load(f)
        
        # print(len(train_features))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_guids = [f.guid for f in train_features]

        id_tensor = torch.tensor(list(range(len(train_features))), dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_sep_ids = torch.tensor([f.sep_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sep_ids, all_label_ids, id_tensor)
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
                input_ids, input_mask, segment_ids, sep_ids, label_ids, id_ = batch
                if not args.use_ft:
                    output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids)
                    loss = output.loss
                if args.use_ft:
                    ##### triple
                    logits_triple = model_field(0, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    logits_start = model_field(1, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    logits_end = model_field(2, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    paths = [logits_triple, logits_start, logits_end]
                    # print(paths)
                    for z in range(args.max_path_num):
                        logits_path = model_field(3+z, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                        paths.append(logits_path)

                    f_tensor = torch.cat(paths, dim = -1)
                    # f_tensor = torch.reshape(f_tensor, [-1,8,768] )
                    # f_tensor = torch.sum(f_tensor, dim = 1)
                    output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids, use_ftcls=True, ft_tensor = f_tensor)
                    loss = output.loss
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
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
                        _save_ckpt(args, model, tokenizer, output_dir = "{}/{}".format(args.output_dir, global_step))

                    # if global_step % args.name_save_steps == 0 and global_step > 1:
                        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                            model_path = find_latest_folder(args.output_dir)
                            model = AutoModelForSequenceClassification.from_pretrained(
                                model_path,
                                from_tf=bool(".ckpt" in args.model_name_or_path),
                                config=config,
                                cache_dir=args.cache_dir,
                                )
                            model.to(device)
                            if args.local_rank != -1:
                                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                                output_device=args.local_rank)
                            elif n_gpu > 1:
                                model = torch.nn.DataParallel(model)
                            eval_examples = processor.get_dev_examples(args.data_dir)
                            if not os.path.exists(os.path.join(args.feature_dir,"dev.pkl")):
                                eval_features = convert_examples_to_features(
                                    eval_examples, args.max_seq_length, tokenizer)
                                with open(os.path.join(args.feature_dir,"dev.pkl"),"wb") as f:
                                    pickle.dump(eval_features, f)
                                eval_features = eval_features[:5000]
                            else:
                                with open(os.path.join(args.feature_dir,"dev.pkl"),"rb") as f:
                                    eval_features = pickle.load(f)
                                eval_features = eval_features[:5000]
                            logger.info("***** Running evaluation *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                            logger.info("  Batch size = %d", args.eval_batch_size)
                            eval_guids = [f.guid for f in eval_features]

                            eval_id_tensor = torch.tensor(list(range(len(eval_features))), dtype=torch.long)
                            eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                            eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                            eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                            eval_sep_ids = torch.tensor([f.sep_ids for f in eval_features], dtype=torch.long)
                            eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                            eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_sep_ids, eval_label_ids, eval_id_tensor)
                            # Run prediction for full data
                            eval_sampler = SequentialSampler(eval_data)
                            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                            model.eval()
                            eval_loss, eval_accuracy = 0, 0
                            nb_eval_steps, nb_eval_examples = 0, 0
                            output = None
                            for input_ids, input_mask, segment_ids, sep_ids, label_ids, id_ in eval_dataloader:
                                input_ids = input_ids.to(device)
                                input_mask = input_mask.to(device)
                                segment_ids = segment_ids.to(device)
                                sep_ids = sep_ids.to(device)
                                label_ids = label_ids.to(device)
                                id_ = id_.to(device)

                                with torch.no_grad():
                                    # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                                    # logits = model(input_ids, segment_ids, input_mask)
                                    # output = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids)
                                    if not args.use_ft:
                                        output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids)
                                        loss = output.loss
                                    if args.use_ft:
                                        ##### triple
                                        logits_triple = model_field(0, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                                        logits_start = model_field(1, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                                        logits_end = model_field(2, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                                        paths = [logits_triple, logits_start, logits_end]
                                        for z in range(args.max_path_num):
                                            logits_path = model_field(3+z, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                                            paths.append(logits_path)

                                        f_tensor = torch.cat(paths, dim = -1)
                                        # f_tensor = torch.reshape(f_tensor, [-1,8,768] )
                                        # f_tensor = torch.sum(f_tensor, dim = 1)
                                        output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids, use_ftcls=True, ft_tensor = f_tensor)
                                        loss = output.loss
                                    tmp_eval_loss = output.loss
                                    logits = output.logits

                                logits = logits.detach().cpu().numpy()
                                label_ids = label_ids.to('cpu').numpy()
                                tmp_eval_accuracy = accuracy(logits, label_ids)

                                eval_loss += tmp_eval_loss.mean().item()
                                eval_accuracy += tmp_eval_accuracy

                                nb_eval_examples += input_ids.size(0)
                                nb_eval_steps += 1

                            eval_loss = eval_loss / nb_eval_steps
                            eval_accuracy = eval_accuracy / nb_eval_examples
                            model.train()
                            if eval_accuracy > best_model["eval_result"]:
                                best_model["eval_result"] = eval_accuracy
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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            )
        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        test_examples = processor.get_test_examples(args.data_dir)
        if not os.path.exists(os.path.join(args.feature_dir,"test.pkl")):
            test_features = convert_examples_to_features(
                test_examples, args.max_seq_length, tokenizer)
            with open(os.path.join(args.feature_dir,"test.pkl"),"wb") as f:
                pickle.dump(test_features, f)
        else:
            with open(os.path.join(args.feature_dir, "test.pkl"),"rb") as f:
                test_features = pickle.load(f)
        logger.info("***** Running Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        test_guids = [f.guid for f in test_features]

        test_id_tensor = torch.tensor(list(range(len(test_features))), dtype=torch.long)        
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_sep_ids = torch.tensor([f.sep_ids for f in test_features], dtype=torch.long)
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_sep_ids, test_label_ids, test_id_tensor)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        output = None
        predict_scores = []
        for input_ids, input_mask, segment_ids, sep_ids, label_ids, id_ in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            sep_ids = sep_ids.to(device)
            label_ids = label_ids.to(device)
            id_ = id_.to(device)

            with torch.no_grad():
                # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                # logits = model(input_ids, segment_ids, input_mask)
                # output = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids)
                if not args.use_ft:
                    output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids)
                    loss = output.loss
                if args.use_ft:
                    ##### triple
                    logits_triple = model_field(0, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    logits_start = model_field(1, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    logits_end = model_field(2, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                    paths = [logits_triple, logits_start, logits_end]
                    for z in range(args.max_path_num):
                        logits_path = model_field(3+z, input_ids, label_ids, model, sep_ids, input_mask, segment_ids)
                        paths.append(logits_path)

                    f_tensor = torch.cat(paths, dim = -1)
                    # f_tensor = torch.reshape(f_tensor, [-1,8,768] )
                    # f_tensor = torch.sum(f_tensor, dim = 1)
                    output, _ = model(input_ids, attention_mask = input_mask, position_ids = segment_ids, labels = label_ids, use_ftcls=True, ft_tensor = f_tensor)
                    loss = output.loss
                tmp_eval_loss = output.loss
                logits = output.logits

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            predict_scores.append([logits, label_ids, id_.cpu().numpy()])

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        # if eval_accuracy > best_model["eval_result"]:
        best_model["test_result"] = eval_accuracy
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
