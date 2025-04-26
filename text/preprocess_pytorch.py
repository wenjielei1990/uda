# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocessing for text classifications. PyTorch implementation."""

import argparse
import copy
import json
import os
import logging
import numpy as np
import torch
import random
from tqdm import tqdm

# Import PyTorch versions of augmentation modules
from augmentation import sent_level_augment_pytorch as sent_level_augment
from augmentation import word_level_augment_pytorch as word_level_augment
from utils import raw_data_utils
from utils import tokenization


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_data_for_worker(examples, replicas, worker_id):
    """Split data among workers for parallel processing."""
    data_per_worker = len(examples) // replicas
    remainder = len(examples) - replicas * data_per_worker
    if worker_id < remainder:
        start = (data_per_worker + 1) * worker_id
        end = (data_per_worker + 1) * (worker_id + 1)
    else:
        start = data_per_worker * worker_id + remainder
        end = data_per_worker * (worker_id + 1) + remainder
    if worker_id == replicas - 1:
        assert end == len(examples)
    logger.info(f"processing data from {start} to {end}")
    examples = examples[start: end]
    return examples, start, end


def build_vocab(examples):
    """Build vocabulary from examples."""
    vocab = {}
    
    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)
                
    for i in range(len(examples)):
        add_to_vocab(examples[i].word_list_a)
        if examples[i].text_b:
            add_to_vocab(examples[i].word_list_b)
    return vocab


def get_data_stats(data_stats_dir, sub_set, sup_size, replicas, examples):
    """Get or compute data statistics (TF-IDF) for the dataset."""
    data_stats_dir = f"{data_stats_dir}/{sub_set}"
    keys = ["tf_idf", "idf"]
    all_exist = True
    
    # Check if stats files already exist
    for key in keys:
        data_stats_path = f"{data_stats_dir}/{key}.json"
        if not os.path.exists(data_stats_path):
            all_exist = False
            logger.info(f"Not exist: {data_stats_path}")
    
    # Load existing stats or compute new ones
    if all_exist:
        logger.info(f"loading data stats from {data_stats_dir}")
        data_stats = {}
        for key in keys:
            with open(f"{data_stats_dir}/{key}.json") as inf:
                data_stats[key] = json.load(inf)
    else:
        assert sup_size == -1, "should use the complete set to get tf_idf"
        assert replicas == 1, "should use the complete set to get tf_idf"
        data_stats = word_level_augment.get_data_stats(examples)
        os.makedirs(data_stats_dir, exist_ok=True)
        for key in keys:
            with open(f"{data_stats_dir}/{key}.json", "w") as ouf:
                json.dump(data_stats[key], ouf)
        logger.info(f"dumped data stats to {data_stats_dir}")
    
    return data_stats


def tokenize_examples(examples, tokenizer):
    """Tokenize examples using the provided tokenizer."""
    logger.info("tokenizing examples")
    for i in tqdm(range(len(examples))):
        examples[i].word_list_a = tokenizer.tokenize_to_word(examples[i].text_a)
        if examples[i].text_b:
            examples[i].word_list_b = tokenizer.tokenize_to_word(examples[i].text_b)
    return examples


def _truncate_seq_pair_keep_right(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length, keeping right tokens."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop(0)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

    def get_dict_features(self):
        return {
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "input_type_ids": self.input_type_ids,
            "label_ids": [self.label_id]
        }


class PairedUnsupInputFeatures(object):
    """Features for paired unsup data."""

    def __init__(self, ori_input_ids, ori_input_mask, ori_input_type_ids,
                aug_input_ids, aug_input_mask, aug_input_type_ids):
        self.ori_input_ids = ori_input_ids
        self.ori_input_mask = ori_input_mask
        self.ori_input_type_ids = ori_input_type_ids
        self.aug_input_ids = aug_input_ids
        self.aug_input_mask = aug_input_mask
        self.aug_input_type_ids = aug_input_type_ids

    def get_dict_features(self):
        return {
            "ori_input_ids": self.ori_input_ids,
            "ori_input_mask": self.ori_input_mask,
            "ori_input_type_ids": self.ori_input_type_ids,
            "aug_input_ids": self.aug_input_ids,
            "aug_input_mask": self.aug_input_mask,
            "aug_input_type_ids": self.aug_input_type_ids,
        }


def convert_examples_to_features(
    examples, label_list, seq_length, tokenizer, trunc_keep_right,
    data_stats=None, aug_ops=None):
    """Convert examples to features that can be used for model training."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    logger.info(f"number of examples to process: {len(examples)}")

    features = []

    if aug_ops:
        logger.info("building vocab")
        word_vocab = build_vocab(examples)
        examples = word_level_augment.word_level_augment(
            examples, aug_ops, word_vocab, data_stats
        )

    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):
        tokens_a = tokenizer.tokenize_to_wordpiece(example.word_list_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize_to_wordpiece(example.word_list_b)

        if tokens_b:
            # Account for [CLS], [SEP], [SEP] with "- 3"
            if trunc_keep_right:
                _truncate_seq_pair_keep_right(tokens_a, tokens_b, seq_length - 3)
            else:
                _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                if trunc_keep_right:
                    tokens_a = tokens_a[-(seq_length - 2):]
                else:
                    tokens_a = tokens_a[0:(seq_length - 2)]

        # Build token sequence with special tokens
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
            logger.info(f"input_type_ids: {' '.join([str(x) for x in input_type_ids])}")
            logger.info(f"label: {example.label} (id = {label_id})")

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))
    
    return features


def dump_tfrecord(features, data_path, worker_id=None, max_shard_size=4096):
    """Save features to files for later use."""
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    logger.info("Saving processed features to files")
    np.random.shuffle(features)
    
    # Split into shards
    shard_cnt = 0
    shard_size = 0
    current_shard = []
    all_shards = []
    
    for feature in features:
        current_shard.append(feature.get_dict_features())
        shard_size += 1
        
        if shard_size >= max_shard_size:
            all_shards.append(current_shard)
            current_shard = []
            shard_size = 0
            shard_cnt += 1
    
    # Don't forget the last shard
    if current_shard:
        all_shards.append(current_shard)
        shard_cnt += 1
    
    # Save each shard to a file
    for i, shard in enumerate(all_shards):
        shard_file = os.path.join(data_path, f"tf_examples.tfrecord.{worker_id if worker_id is not None else 0}.{i}")
        with open(shard_file, 'w') as f:
            json.dump(shard, f)
    
    logger.info(f"Saved {len(features)} examples in {shard_cnt} shards")


def get_data_by_size_lim(train_examples, processor, sup_size):
    """Get a balanced dataset with only sup_size examples."""
    assert sup_size % len(processor.get_labels()) == 0
    per_label_size = sup_size // len(processor.get_labels())
    per_label_examples = {}
    
    for i in range(len(train_examples)):
        label = train_examples[i].label
        if label not in per_label_examples:
            per_label_examples[label] = []
        per_label_examples[label].append(train_examples[i])

    for label in processor.get_labels():
        assert len(per_label_examples[label]) >= per_label_size, (
            f"label {label} only has {len(per_label_examples[label])} examples while the limit "
            f"is {per_label_size}")

    new_train_examples = []
    for i in range(per_label_size):
        for label in processor.get_labels():
            new_train_examples.append(per_label_examples[label][i])
    
    return new_train_examples


def proc_and_save_sup_data(
    processor, sub_set, raw_data_dir, sup_out_dir,
    tokenizer, max_seq_length, trunc_keep_right,
    worker_id, replicas, sup_size):
    """Process and save supervised data."""
    logger.info("Getting examples")
    if sub_set == "train":
        examples = processor.get_train_examples(raw_data_dir)
    elif sub_set == "dev":
        examples = processor.get_dev_examples(raw_data_dir)
        assert replicas == 1, "dev set can be processed with just one worker"
        assert sup_size == -1, "should use the full dev set"

    if sup_size != -1:
        logger.info(f"Setting number of examples to {sup_size}")
        examples = get_data_by_size_lim(
            examples, processor, sup_size)
    
    if replicas != 1:
        if len(examples) < replicas:
            replicas = len(examples)
            if worker_id >= replicas:
                return
        examples, _, _ = get_data_for_worker(
            examples, replicas, worker_id)

    logger.info("Processing data")
    examples = tokenize_examples(examples, tokenizer)

    features = convert_examples_to_features(
        examples, processor.get_labels(), max_seq_length, tokenizer,
        trunc_keep_right, None, None)
    
    dump_tfrecord(features, sup_out_dir, worker_id)


def proc_and_save_unsup_data(
    processor, sub_set,
    raw_data_dir, data_stats_dir, unsup_out_dir,
    tokenizer,
    max_seq_length, trunc_keep_right,
    aug_ops, aug_copy_num,
    worker_id, replicas):
    """Process and save unsupervised data."""
    # Set random seed for reproducibility while still varying between runs
    random_seed = random.randint(0, 100000)
    logger.info(f"random seed: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    logger.info("Getting examples")

    if sub_set == "train":
        ori_examples = processor.get_train_examples(raw_data_dir)
    elif sub_set.startswith("unsup"):
        ori_examples = processor.get_unsup_examples(raw_data_dir, sub_set)
    else:
        assert False, f"Unsupported subset: {sub_set}"
    
    # Save total size before splitting
    data_total_size = len(ori_examples)
    
    if replicas != -1:
        ori_examples, start, end = get_data_for_worker(
            ori_examples, replicas, worker_id)
    else:
        start = 0
        end = len(ori_examples)

    logger.info("Getting augmented examples")
    aug_examples = copy.deepcopy(ori_examples)
    
    # Set up augmentation config
    sent_level_augment.config.back_translation_dir = data_stats_dir
    
    aug_examples = sent_level_augment.run_augment(
        aug_examples, aug_ops, sub_set,
        aug_copy_num,
        start, end, data_total_size)

    labels = processor.get_labels() + ["unsup"]
    logger.info("Processing ori examples")
    ori_examples = tokenize_examples(ori_examples, tokenizer)
    ori_features = convert_examples_to_features(
        ori_examples, labels, max_seq_length, tokenizer,
        trunc_keep_right, None, None)

    if "idf" in aug_ops:
        data_stats = get_data_stats(
            data_stats_dir, sub_set,
            -1, replicas, ori_examples)
    else:
        data_stats = None

    logger.info("Processing aug examples")
    aug_examples = tokenize_examples(aug_examples, tokenizer)
    aug_features = convert_examples_to_features(
        aug_examples, labels, max_seq_length, tokenizer,
        trunc_keep_right, data_stats, aug_ops)

    # Create paired features
    unsup_features = []
    for ori_feat, aug_feat in zip(ori_features, aug_features):
        unsup_features.append(PairedUnsupInputFeatures(
            ori_feat.input_ids,
            ori_feat.input_mask,
            ori_feat.input_type_ids,
            aug_feat.input_ids,
            aug_feat.input_mask,
            aug_feat.input_type_ids,
        ))
    
    dump_tfrecord(unsup_features, unsup_out_dir, worker_id)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument("--task_name", default="IMDB", type=str,
                      help="The name of the task to train.")
    
    # Data directories
    parser.add_argument("--raw_data_dir", type=str, required=True,
                      help="Data directory of the raw data")
    parser.add_argument("--output_base_dir", type=str, required=True,
                      help="Data directory of the processed data")
    
    # Augmentation parameters
    parser.add_argument("--aug_ops", default="bt-0.9", type=str,
                      help="Augmentation method")
    parser.add_argument("--aug_copy_num", type=int, default=-1,
                      help="Index of the generated augmented example")
    
    # Sequence parameters
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length after tokenization")
    parser.add_argument("--sup_size", type=int, default=-1,
                      help="Size of the labeled set")
    parser.add_argument("--trunc_keep_right", type=bool, default=True,
                      help="Whether to keep the right part when truncate a sentence")
    
    # Task type
    parser.add_argument("--data_type", choices=["sup", "unsup"], default="sup",
                      help="Which preprocess task to perform")
    parser.add_argument("--sub_set", default="train", type=str,
                      help="Which sub_set to preprocess")
    
    # BERT parameters
    parser.add_argument("--vocab_file", type=str, required=True,
                      help="The path of the vocab file of BERT")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                      help="Whether to use uncased text for BERT")
    
    # Back translation directory
    parser.add_argument("--back_translation_dir", type=str, default="",
                      help="Directory for back translated sentences")
    
    # Parallel processing
    parser.add_argument("--replicas", type=int, default=1,
                      help="Number of workers for parallel preprocessing")
    parser.add_argument("--worker_id", type=int, default=0,
                      help="Worker ID for parallel preprocessing")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.max_seq_length > 512:
        raise ValueError(
            f"Cannot use sequence length {args.max_seq_length} because the BERT model "
            f"was only trained up to sequence length 512")

    processor = raw_data_utils.get_processor(args.task_name)
    
    # Create tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    if args.data_type == "sup":
        sup_out_dir = args.output_base_dir
        logger.info(f"Create sup. data: subset {args.sub_set} => {sup_out_dir}")

        proc_and_save_sup_data(
            processor, args.sub_set, args.raw_data_dir, sup_out_dir,
            tokenizer, args.max_seq_length, args.trunc_keep_right,
            args.worker_id, args.replicas, args.sup_size,
        )
    elif args.data_type == "unsup":
        assert args.aug_ops is not None, \
            "aug_ops is required to preprocess unsupervised data."
        
        unsup_out_dir = os.path.join(
            args.output_base_dir,
            args.aug_ops,
            str(args.aug_copy_num))
        
        data_stats_dir = args.back_translation_dir
        if not data_stats_dir:
            data_stats_dir = os.path.join(args.raw_data_dir, "data_stats")

        logger.info(f"Create unsup. data: subset {args.sub_set} => {unsup_out_dir}")
        proc_and_save_unsup_data(
            processor, args.sub_set,
            args.raw_data_dir, data_stats_dir, unsup_out_dir,
            tokenizer, args.max_seq_length, args.trunc_keep_right,
            args.aug_ops, args.aug_copy_num,
            args.worker_id, args.replicas)


if __name__ == "__main__":
    main() 