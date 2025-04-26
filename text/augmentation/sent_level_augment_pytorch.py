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
"""Sentence level augmentations: back translation.

PyTorch implementation.
"""

import math
import random
import logging
import os
import numpy as np
import torch

from augmentation import word_level_augment_pytorch as word_level_augment


class AugmentationConfig:
    """Configuration for augmentation."""
    def __init__(self):
        self.back_translation_dir = None


# Global configuration object that can be set by the main application
config = AugmentationConfig()


def replace_with_length_check(
    ori_text, new_text,
    use_min_length,
    use_max_length_diff_ratio):
    """Use new_text if the text length satisfies several constraints."""
    if len(ori_text) < use_min_length or len(new_text) < use_min_length:
        if random.random() < 0.001:
            logging.info(
                "not replacing due to short text: \n\tori: %s\n\tnew: %s\n",
                word_level_augment.filter_unicode(ori_text),
                word_level_augment.filter_unicode(new_text))
        return ori_text
    length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
    if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
        if random.random() < 0.001:
            logging.info(
                "not replacing due to too different text length:\n"
                "\tori: %s\n\tnew: %s\n",
                word_level_augment.filter_unicode(ori_text),
                word_level_augment.filter_unicode(new_text))
        return ori_text
    return new_text


def back_translation(examples, aug_ops, sub_set, aug_copy_num,
                     start, end, data_total_size):
    """Run back translation."""
    use_min_length = 10
    use_max_length_diff_ratio = 0.5
    logging.info("running bt augmentation")
    bt_args = aug_ops.split("-")
    temp = float(bt_args[1])

    if len(bt_args) > 2:
        assert len(bt_args) == 3
        assert float(bt_args[2]) == 1.

    if examples[0].text_b is not None:
        text_per_example = 2
    else:
        text_per_example = 1

    back_translation_file = f"{config.back_translation_dir}/{sub_set}/sample_{temp:.1f}/para/para_{aug_copy_num}.txt"
    logging.info(f"Using back translation file: {back_translation_file}")

    with open(back_translation_file, 'r') as inf:
        paraphrases = inf.readlines()
    for i in range(len(paraphrases)):
        paraphrases[i] = paraphrases[i].strip()
    assert len(paraphrases) == data_total_size

    paraphrases = paraphrases[start * text_per_example : end * text_per_example]
    aug_examples = []
    aug_cnt = 0
    for i in range(len(examples)):
        ori_example = examples[i]
        text_a = replace_with_length_check(
            ori_example.text_a,
            paraphrases[i * text_per_example],
            use_min_length,
            use_max_length_diff_ratio,
        )
        if text_a == paraphrases[i * text_per_example]:
            aug_cnt += 1
        if ori_example.text_b is not None:
            text_b = replace_with_length_check(
                ori_example.text_b,
                paraphrases[i * text_per_example + 1],
                use_min_length,
                use_max_length_diff_ratio,
            )
        else:
            text_b = None

        # Assume InputExample class is defined elsewhere in your codebase
        example = type(ori_example)(
            guid=ori_example.guid,
            text_a=text_a,
            text_b=text_b,
            label=ori_example.label)
        aug_examples.append(example)
        if np.random.random() < 0.0001:
            logging.info("\tori:\n\t\t%s\n\t\t%s\n\t\t%s\n",
                ori_example.text_a, ori_example.text_b, ori_example.label)
            logging.info("\tnew:\n\t\t%s\n\t\t%s\n\t\t%s\n",
                example.text_a, example.text_b, example.label)
        if i % 10000 == 0:
            print(f"processing example # {i}")
    logging.info("applied back translation for %.1f percent of data",
            aug_cnt * 1. / len(examples) * 100)
    logging.info("finishing running back translation augmentation")
    return aug_examples


def run_augment(
    examples, aug_ops, sub_set, aug_copy_num,
    start, end, dst_tot_size):
    """Sentence level augmentations. Used before augmentation."""
    if aug_ops:
        if aug_ops.startswith("bt"):
            examples = back_translation(
                examples, aug_ops, sub_set, aug_copy_num, start, end, dst_tot_size)
        else:
            pass
    return examples 