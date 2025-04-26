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
"""Build datasets for PyTorch implementation.
"""

import collections
import copy
import json
import os
import logging
import glob
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """Features for supervised data."""
    
    def __init__(self, input_ids, input_mask, input_type_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_ids = label_ids
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "input_type_ids": self.input_type_ids,
            "label_ids": self.label_ids
        }


class UnsupInputFeatures(object):
    """Features for unsupervised data."""
    
    def __init__(self, ori_input_ids, ori_input_mask, ori_input_type_ids,
                aug_input_ids, aug_input_mask, aug_input_type_ids):
        self.ori_input_ids = ori_input_ids
        self.ori_input_mask = ori_input_mask
        self.ori_input_type_ids = ori_input_type_ids
        self.aug_input_ids = aug_input_ids
        self.aug_input_mask = aug_input_mask
        self.aug_input_type_ids = aug_input_type_ids
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "ori_input_ids": self.ori_input_ids,
            "ori_input_mask": self.ori_input_mask,
            "ori_input_type_ids": self.ori_input_type_ids,
            "aug_input_ids": self.aug_input_ids,
            "aug_input_mask": self.aug_input_mask,
            "aug_input_type_ids": self.aug_input_type_ids
        }


def load_and_cache_examples(data_path: str) -> List[Dict]:
    """Load examples from cached files."""
    all_examples = []
    
    # Find all tfrecord files in the directory
    file_pattern = os.path.join(data_path, "tf_examples.tfrecord*")
    for file_path in glob.glob(file_pattern):
        try:
            with open(file_path, 'r') as f:
                examples = json.load(f)
                all_examples.extend(examples)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    # Shuffle examples
    random.shuffle(all_examples)
    return all_examples


def get_aug_files(data_base_path: str, aug_ops: str, aug_copy: int) -> List[str]:
    """Get augmentation files."""
    sub_policy_list = aug_ops.split("+")
    total_data_files = []
    
    for sub_policy in sub_policy_list:
        sub_policy_data_files = []
        exist_copy_num = {}
        
        sub_policy_dir = os.path.join(data_base_path, sub_policy)
        if not os.path.exists(sub_policy_dir):
            logger.warning(f"Subpolicy directory not found: {sub_policy_dir}")
            continue
            
        for copy_dir in os.listdir(sub_policy_dir):
            try:
                copy_num = int(copy_dir.strip("/"))
                if copy_num >= aug_copy:
                    continue
                    
                exist_copy_num[copy_num] = 1
                data_record_path = os.path.join(
                    data_base_path, sub_policy, copy_dir, "tf_examples.tfrecord*")
                data_files = glob.glob(data_record_path)
                sub_policy_data_files.extend(data_files)
            except (ValueError, OSError) as e:
                logger.warning(f"Error processing {copy_dir}: {e}")
        
        if len(exist_copy_num) < aug_copy * 0.9:
            logger.warning(f"Not enough copies for aug op: {aug_ops}")
            logger.warning(f"Found files: {' '.join(sub_policy_data_files)}")
            logger.warning(f"Found copy: {len(exist_copy_num)} / desired copy: {aug_copy}")
        
        assert len(exist_copy_num) > aug_copy * 0.9, f"Not enough copies for {sub_policy}"
        total_data_files.extend(sub_policy_data_files)
    
    # Shuffle files
    random.shuffle(total_data_files)
    return total_data_files


class SupDataset(Dataset):
    """Dataset for supervised data."""
    
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        features = {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "input_mask": torch.tensor(example["input_mask"], dtype=torch.long),
            "input_type_ids": torch.tensor(example["input_type_ids"], dtype=torch.long),
            "label_ids": torch.tensor(example["label_ids"], dtype=torch.long)
        }
        
        return features


class UnsupDataset(Dataset):
    """Dataset for unsupervised data."""
    
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        features = {
            "ori_input_ids": torch.tensor(example["ori_input_ids"], dtype=torch.long),
            "ori_input_mask": torch.tensor(example["ori_input_mask"], dtype=torch.long),
            "ori_input_type_ids": torch.tensor(example["ori_input_type_ids"], dtype=torch.long),
            "aug_input_ids": torch.tensor(example["aug_input_ids"], dtype=torch.long),
            "aug_input_mask": torch.tensor(example["aug_input_mask"], dtype=torch.long),
            "aug_input_type_ids": torch.tensor(example["aug_input_type_ids"], dtype=torch.long)
        }
        
        return features


class CombinedDataset(Dataset):
    """Dataset that combines supervised and unsupervised data."""
    
    def __init__(self, sup_dataset, unsup_dataset, unsup_ratio):
        self.sup_dataset = sup_dataset
        self.unsup_dataset = unsup_dataset
        self.unsup_ratio = unsup_ratio
        
        # Effective length is determined by the supervised data
        self.length = len(sup_dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get supervised example
        sup_example = self.sup_dataset[idx % len(self.sup_dataset)]
        
        # Get unsupervised example if available
        if self.unsup_dataset is not None and len(self.unsup_dataset) > 0:
            unsup_idx = idx % len(self.unsup_dataset)
            unsup_example = self.unsup_dataset[unsup_idx]
            
            # Combine features
            features = {**sup_example, **unsup_example}
        else:
            features = sup_example
            
        return features


def get_sup_eval_dataloader(data_path: str, batch_size: int, max_seq_length: int):
    """Get evaluation dataloader for supervised data."""
    # Load examples
    examples = load_and_cache_examples(data_path)
    
    # Create dataset
    dataset = SupDataset(examples)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )
    
    return dataloader


def get_train_dataloader(
    sup_data_path: Optional[str] = None,
    unsup_data_path: Optional[str] = None,
    aug_ops: Optional[str] = None,
    aug_copy: Optional[int] = None,
    unsup_ratio: Optional[int] = None,
    batch_size: int = 32,
    max_seq_length: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Get training dataloaders."""
    
    sup_dataset = None
    unsup_dataset = None
    
    # Load supervised data if provided
    if sup_data_path:
        sup_examples = load_and_cache_examples(sup_data_path)
        sup_dataset = SupDataset(sup_examples)
        logger.info(f"Loaded {len(sup_dataset)} supervised examples")
    
    # Load unsupervised data if provided and unsup_ratio > 0
    if unsup_data_path and unsup_ratio and unsup_ratio > 0:
        assert aug_ops is not None and aug_copy is not None, \
            "Require aug_ops, aug_copy to load augmented unsup data."
            
        # Get augmentation files
        unsup_files = get_aug_files(unsup_data_path, aug_ops, aug_copy)
        
        # Load examples from all files
        unsup_examples = []
        for file_path in unsup_files:
            try:
                with open(file_path, 'r') as f:
                    file_examples = json.load(f)
                    unsup_examples.extend(file_examples)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        # Shuffle examples
        random.shuffle(unsup_examples)
        
        # Create unsupervised dataset
        unsup_dataset = UnsupDataset(unsup_examples)
        logger.info(f"Loaded {len(unsup_dataset)} unsupervised examples")
    
    # Create combined dataset if both supervised and unsupervised data are available
    if sup_dataset and unsup_dataset:
        combined_dataset = CombinedDataset(sup_dataset, unsup_dataset, unsup_ratio)
        
        # Create dataloader
        dataloader = DataLoader(
            combined_dataset,
            sampler=RandomSampler(combined_dataset),
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        return dataloader
    
    # If only supervised data is available
    elif sup_dataset:
        dataloader = DataLoader(
            sup_dataset,
            sampler=RandomSampler(sup_dataset),
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        return dataloader
    
    # If no data is available
    else:
        logger.warning("No data provided for training")
        return None 