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
"""Runner for UDA that uses BERT. PyTorch implementation."""

import argparse
import json
import logging
import os
import torch
from torch.utils.data import Dataset

from text.bert.modeling_pytorch import BertConfig
import text.uda_pytorch as uda
from text.uda_pytorch import UDAConfig, UDAModel, UDATrainer, create_uda_model


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch dataset for text data."""
    
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i):
        feature = self.features[i]
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["input_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(feature["input_type_ids"], dtype=torch.long),
        }
        
        if "label_ids" in feature:
            item["labels"] = torch.tensor(feature["label_ids"][0], dtype=torch.long)
            
        return item


class UnsupPairedDataset(Dataset):
    """PyTorch dataset for paired unsupervised data."""
    
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i):
        feature = self.features[i]
        
        return {
            "ori_input_ids": torch.tensor(feature["ori_input_ids"], dtype=torch.long),
            "ori_attention_mask": torch.tensor(feature["ori_input_mask"], dtype=torch.long),
            "ori_token_type_ids": torch.tensor(feature["ori_input_type_ids"], dtype=torch.long),
            "aug_input_ids": torch.tensor(feature["aug_input_ids"], dtype=torch.long),
            "aug_attention_mask": torch.tensor(feature["aug_input_mask"], dtype=torch.long),
            "aug_token_type_ids": torch.tensor(feature["aug_input_type_ids"], dtype=torch.long),
        }


def load_tf_examples(file_path):
    """Load examples from TensorFlow TFRecord files."""
    import tensorflow as tf
    
    all_features = []
    
    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([512], tf.int64),
            "input_mask": tf.io.FixedLenFeature([512], tf.int64),
            "input_type_ids": tf.io.FixedLenFeature([512], tf.int64),
        }
        
        # For supervised data
        if "sup" in file_path:
            name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        # For unsupervised data
        else:
            name_to_features.update({
                "ori_input_ids": tf.io.FixedLenFeature([512], tf.int64),
                "ori_input_mask": tf.io.FixedLenFeature([512], tf.int64),
                "ori_input_type_ids": tf.io.FixedLenFeature([512], tf.int64),
                "aug_input_ids": tf.io.FixedLenFeature([512], tf.int64),
                "aug_input_mask": tf.io.FixedLenFeature([512], tf.int64),
                "aug_input_type_ids": tf.io.FixedLenFeature([512], tf.int64),
            })
            
        example = tf.io.parse_single_example(record, name_to_features)
        return example
    
    # Find all TFRecord files in directory
    tf_record_pattern = os.path.join(file_path, "tf_examples.tfrecord*")
    for tf_record_file in tf.io.gfile.glob(tf_record_pattern):
        logger.info(f"Loading TFRecord file: {tf_record_file}")
        dataset = tf.data.TFRecordDataset(tf_record_file)
        
        for record in dataset:
            example = _decode_record(record)
            # Convert to Python dict
            feature = {k: v.numpy().tolist() for k, v in example.items()}
            all_features.append(feature)
        
    return all_features


def get_dataset(data_dir):
    """Load dataset from TFRecord files."""
    if "unsup" in data_dir:
        features = load_tf_examples(data_dir)
        return UnsupPairedDataset(features)
    else:
        features = load_tf_examples(data_dir)
        return TextDataset(features)


def get_label_count(processor_name):
    """Get number of labels for the specified task."""
    label_counts = {
        "CoLA": 2,
        "MNLI": 3,
        "MRPC": 2,
        "SST-2": 2,
        "IMDB": 2,
        "Yelp-2": 2,
        "Yelp-5": 5,
        "Amazon-2": 2,
        "Amazon-5": 5,
    }
    
    if processor_name not in label_counts:
        raise ValueError(f"Unsupported task: {processor_name}")
    
    return label_counts[processor_name]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Task parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    
    # Unsupervised objective parameters
    parser.add_argument("--unsup_ratio", type=int, default=0, help="Ratio of unsupervised batch size to supervised batch size.")
    parser.add_argument("--aug_ops", type=str, default="", help="Augmentation operations.")
    parser.add_argument("--aug_copy", type=int, default=-1, help="Number of different augmented data generated.")
    parser.add_argument("--uda_coeff", type=float, default=1, help="Coefficient on the UDA loss.")
    parser.add_argument("--tsa", type=str, default="", help="TSA scheme.")
    parser.add_argument("--uda_softmax_temp", type=float, default=-1, help="Temperature for soft targets.")
    parser.add_argument("--uda_confidence_thresh", type=float, default=-1, help="Threshold for high-confidence predictions.")
    
    # Model configuration
    parser.add_argument("--bert_config_file", type=str, required=True, help="BERT configuration file.")
    parser.add_argument("--vocab_file", type=str, required=True, help="The vocabulary file BERT was pretrained on.")
    parser.add_argument("--init_checkpoint", type=str, help="Initial checkpoint from pre-trained BERT model.")
    parser.add_argument("--task_name", type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory for output model files.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum total sequence length after tokenization.")
    parser.add_argument("--model_dropout", type=float, default=-1, help="Dropout rate for model.")
    
    # Data directories
    parser.add_argument("--sup_train_data_dir", type=str, help="The input data dir for supervised training.")
    parser.add_argument("--eval_data_dir", type=str, help="The input data dir for evaluation.")
    parser.add_argument("--unsup_data_dir", type=str, help="The input data dir for unsupervised data.")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for supervised training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_train_steps", type=int, required=True, help="Number of training steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every this many steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every this many steps.")
    
    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--num_warmup_steps", type=int, help="Number of warmup steps for optimizer.")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="Gradient clip norm.")
    
    # Device parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run on (cuda/cpu).")
    
    return parser.parse_args()


def main():
    """Main function to run UDA training or evaluation."""
    args = parse_args()
    
    # Create directories
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    # Save args to file
    with open(os.path.join(args.model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set up UDA configuration
    uda.config.unsup_ratio = args.unsup_ratio
    uda.config.uda_coeff = args.uda_coeff
    uda.config.tsa = args.tsa
    uda.config.uda_softmax_temp = args.uda_softmax_temp
    uda.config.uda_confidence_thresh = args.uda_confidence_thresh
    
    # Set device
    device = torch.device(args.device)
    
    # Get label count for the task
    num_labels = get_label_count(args.task_name)
    
    # Load BERT config
    bert_config = BertConfig.from_json_file(
        args.bert_config_file, 
        model_dropout=args.model_dropout if args.model_dropout != -1 else None
    )
    
    # Create model
    model = create_uda_model(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=args.init_checkpoint
    )
    
    # Log model info
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load datasets
    train_dataset = None
    eval_dataset = None
    unsup_dataset = None
    
    if args.do_train:
        logger.info(f"Loading supervised training data from {args.sup_train_data_dir}")
        train_dataset = get_dataset(args.sup_train_data_dir)
        
        if args.unsup_ratio > 0 and args.unsup_data_dir:
            logger.info(f"Loading unsupervised data from {args.unsup_data_dir}")
            unsup_dataset = get_dataset(args.unsup_data_dir)
    
    if args.do_eval and args.eval_data_dir:
        logger.info(f"Loading evaluation data from {args.eval_data_dir}")
        eval_dataset = get_dataset(args.eval_data_dir)
    
    # Set up trainer
    if args.num_warmup_steps is None:
        args.num_warmup_steps = args.num_train_steps // 10
        
    logger.info(f"Warmup steps: {args.num_warmup_steps}/{args.num_train_steps}")
    
    trainer = UDATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        unsup_dataset=unsup_dataset,
        device=device,
        num_train_steps=args.num_train_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.num_warmup_steps,
        logging_dir=args.model_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        unsup_ratio=args.unsup_ratio,
        uda_coeff=args.uda_coeff,
        tsa=args.tsa,
        clip_norm=args.clip_norm,
    )
    
    # Training
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info(f"  Supervised batch size = {args.train_batch_size}")
        logger.info(f"  Unsupervised batch size = {args.train_batch_size * args.unsup_ratio}")
        logger.info(f"  Num steps = {args.num_train_steps}")
        trainer.train()
    
    # Evaluation
    if args.do_eval and eval_dataset:
        logger.info("***** Running evaluation *****")
        logger.info(f"  Batch size = {args.eval_batch_size}")
        
        # If we trained the model, evaluate the final model
        if args.do_train:
            metrics = trainer.evaluate()
            logger.info("Final evaluation results:")
            for key, value in metrics.items():
                logger.info(f"  {key} = {value:.4f}")
        # Otherwise, evaluate all checkpoints
        else:
            checkpoints = list(
                os.path.dirname(c) for c in 
                sorted(glob.glob(os.path.join(args.model_dir, "checkpoint-*", "model.pt")))
            )
            
            best_acc = 0
            for checkpoint in checkpoints:
                logger.info(f"Evaluating checkpoint {checkpoint}")
                
                # Load model weights
                model.load_state_dict(
                    torch.load(os.path.join(checkpoint, "model.pt"), map_location=device)
                )
                
                # Evaluate
                metrics = trainer.evaluate()
                
                # Log metrics
                for key, value in metrics.items():
                    logger.info(f"  {key} = {value:.4f}")
                
                # Update best accuracy
                if "eval_accuracy" in metrics:
                    best_acc = max(best_acc, metrics["eval_accuracy"])
            
            logger.info(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main() 