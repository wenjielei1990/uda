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
"""Code for using the labeled examples and unlabeled examples in unsupervised data augmentation.

PyTorch implementation.
"""

import collections
import re
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from text.bert.modeling_pytorch import BertModel
from text.bert.optimization_pytorch import create_optimizer


class UDAConfig:
    """Configuration for UDA model."""
    def __init__(self):
        # Unsupervised objective related hyperparameters
        self.unsup_ratio = 0
        self.uda_coeff = 1
        self.tsa = ""
        self.uda_softmax_temp = -1
        self.uda_confidence_thresh = -1


# Global configuration object that can be set by the main application
config = UDAConfig()


def kl_for_log_probs(log_p, log_q):
    """KL divergence between log-probabilities."""
    p = torch.exp(log_p)
    neg_ent = torch.sum(p * log_p, dim=-1)
    neg_cross_ent = torch.sum(p * log_q, dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl


class ClassificationHead(nn.Module):
    """Classification head for BERT model."""
    
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, is_training=True):
        if is_training:
            hidden_states = self.dropout(hidden_states)
        logits = self.dense(hidden_states)
        return logits


def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    """Get the training signal annealing (TSA) threshold."""
    training_progress = torch.tensor(global_step, dtype=torch.float32) / torch.tensor(num_train_steps, dtype=torch.float32)
    
    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - torch.exp((-training_progress) * scale)
    else:
        threshold = torch.tensor(1.0)
        
    return threshold * (end - start) + start


class UDAModel(nn.Module):
    """UDA model implementation."""
    
    def __init__(self, bert_model, num_labels):
        super(UDAModel, self).__init__()
        self.bert = bert_model
        self.classifier = ClassificationHead(
            bert_model.config.hidden_size, 
            num_labels,
            dropout_prob=bert_model.config.hidden_dropout_prob
        )
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        unsup_ratio=0,
        tsa=None,
        global_step=None,
        num_train_steps=None,
        ori_input_ids=None,
        ori_attention_mask=None,
        ori_token_type_ids=None,
        aug_input_ids=None,
        aug_attention_mask=None,
        aug_token_type_ids=None,
    ):
        """Forward pass with both supervised and unsupervised objectives."""
        device = input_ids.device if input_ids is not None else torch.device("cpu")
        batch_size = input_ids.shape[0] if input_ids is not None else 0
        
        # Determine sizes for supervised and unsupervised batches
        if self.training and unsup_ratio > 0 and ori_input_ids is not None:
            assert batch_size % (1 + 2 * unsup_ratio) == 0
            sup_batch_size = batch_size // (1 + 2 * unsup_ratio)
            unsup_batch_size = sup_batch_size * unsup_ratio
        else:
            sup_batch_size = batch_size
            unsup_batch_size = 0
            
        # Process all inputs
        if unsup_ratio > 0 and ori_input_ids is not None:
            # Combine supervised and unsupervised inputs
            combined_input_ids = torch.cat([
                input_ids, 
                ori_input_ids, 
                aug_input_ids
            ], dim=0)
            
            combined_attention_mask = torch.cat([
                attention_mask, 
                ori_attention_mask, 
                aug_attention_mask
            ], dim=0)
            
            combined_token_type_ids = torch.cat([
                token_type_ids, 
                ori_token_type_ids, 
                aug_token_type_ids
            ], dim=0)
            
            # Get pooled output from BERT
            pooled_output = self.bert(
                input_ids=combined_input_ids,
                token_type_ids=combined_token_type_ids,
                attention_mask=combined_attention_mask,
                output_type="pooled"
            )
            
            # Get logits
            logits = self.classifier(pooled_output, self.training)
            
            # Split outputs
            sup_logits = logits[:sup_batch_size]
            ori_logits = logits[sup_batch_size:sup_batch_size+unsup_batch_size]
            aug_logits = logits[sup_batch_size+unsup_batch_size:]
        else:
            # Process only supervised inputs
            pooled_output = self.bert(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_type="pooled"
            )
            
            logits = self.classifier(pooled_output, self.training)
            sup_logits = logits
            ori_logits = None
            aug_logits = None
        
        # Calculate supervised loss
        loss_dict = {}
        if labels is not None:
            log_probs = F.log_softmax(sup_logits, dim=-1)
            one_hot_labels = F.one_hot(
                labels, num_classes=sup_logits.shape[-1]
            ).to(device=device, dtype=torch.float32)
            
            per_example_loss = -torch.sum(one_hot_labels * log_probs, dim=-1)
            loss_mask = torch.ones_like(per_example_loss)
            
            # TSA masking
            if tsa:
                tsa_start = 1. / sup_logits.shape[-1]
                tsa_threshold = get_tsa_threshold(
                    tsa, global_step, num_train_steps,
                    tsa_start, end=1.0
                ).to(device)
                
                correct_label_probs = torch.sum(one_hot_labels * torch.exp(log_probs), dim=-1)
                larger_than_threshold = correct_label_probs > tsa_threshold
                loss_mask = loss_mask * (1 - larger_than_threshold.to(torch.float32))
            
            loss_mask = loss_mask.detach()  # Stop gradient
            per_example_loss = per_example_loss * loss_mask
            sup_loss = torch.sum(per_example_loss) / torch.maximum(
                torch.sum(loss_mask), torch.tensor(1.0).to(device)
            )
            
            loss_dict["sup_loss"] = sup_loss
            
            # Calculate accuracy metrics
            predictions = torch.argmax(sup_logits, dim=-1)
            is_correct = (predictions == labels).to(torch.float32)
            acc = torch.mean(is_correct)
            loss_dict["sup_acc"] = acc
            
            if tsa:
                loss_dict["tsa_threshold"] = tsa_threshold
                loss_dict["sup_trained_ratio"] = torch.mean(loss_mask)
            
            total_loss = sup_loss
        else:
            total_loss = torch.tensor(0.0).to(device)
        
        # Calculate unsupervised loss
        unsup_loss_mask = None
        if unsup_ratio > 0 and config.uda_coeff > 0 and ori_logits is not None:
            # Get log probabilities
            ori_log_probs = F.log_softmax(ori_logits, dim=-1)
            aug_log_probs = F.log_softmax(aug_logits, dim=-1)
            
            # Optionally use temperature for sharpening predictions
            if config.uda_softmax_temp != -1:
                temp = config.uda_softmax_temp
                tgt_ori_log_probs = F.log_softmax(ori_logits / temp, dim=-1)
                tgt_ori_log_probs = tgt_ori_log_probs.detach()  # Stop gradient
            else:
                tgt_ori_log_probs = ori_log_probs.detach()  # Stop gradient
            
            # Optionally filter examples based on confidence threshold
            unsup_loss_mask = torch.tensor(1.0).to(device)
            if config.uda_confidence_thresh != -1:
                largest_prob = torch.max(torch.exp(ori_log_probs), dim=-1)[0]
                unsup_loss_mask = (
                    largest_prob > config.uda_confidence_thresh
                ).to(torch.float32)
                unsup_loss_mask = unsup_loss_mask.detach()  # Stop gradient
            
            # Calculate KL divergence loss
            per_example_kl_loss = kl_for_log_probs(
                tgt_ori_log_probs, aug_log_probs
            ) * unsup_loss_mask
            
            unsup_loss = torch.mean(per_example_kl_loss)
            loss_dict["unsup_loss"] = unsup_loss
            
            if unsup_loss_mask is not None and unsup_loss_mask.dim() > 0:
                loss_dict["high_prob_ratio"] = torch.mean(unsup_loss_mask)
            
            total_loss = total_loss + config.uda_coeff * unsup_loss
        
        # Add final total loss
        loss_dict["total_loss"] = total_loss
        
        return (total_loss, sup_logits, loss_dict)


class UDATrainer:
    """Trainer for UDA model."""
    
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        unsup_dataset=None,
        device=None,
        num_train_steps=None,
        learning_rate=2e-5,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir=None,
        logging_steps=100,
        save_steps=1000,
        unsup_ratio=0,
        uda_coeff=1,
        tsa=None,
        clip_norm=1.0,
    ):
        """Initialize the trainer."""
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.unsup_dataset = unsup_dataset
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.num_train_steps = num_train_steps
        self.unsup_ratio = unsup_ratio
        self.uda_coeff = uda_coeff
        self.tsa = tsa
        
        # Create optimizer and scheduler
        self.optimizer, self.scheduler, self.grad_clipper = create_optimizer(
            model=model,
            loss=None,  # Not used in our implementation
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=warmup_steps,
            clip_norm=clip_norm
        )
        
        # Setup logging
        self.logging_dir = logging_dir
        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logging_dir)
        else:
            self.writer = None
            
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.global_step = 0
        
    def train(self):
        """Train the model for the specified number of steps."""
        # Setup data loaders
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=32  # This should be configurable
        )
        
        unsup_dataloader = None
        if self.unsup_dataset and self.unsup_ratio > 0:
            unsup_sampler = RandomSampler(self.unsup_dataset)
            unsup_dataloader = DataLoader(
                self.unsup_dataset,
                sampler=unsup_sampler,
                batch_size=32 * self.unsup_ratio  # This should be configurable
            )
            
        # Train loop
        self.model.train()
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(self.train_dataset)}")
        logging.info(f"  Num epochs = (dynamic)")
        logging.info(f"  Total train batch size = {32}")
        logging.info(f"  Total optimization steps = {self.num_train_steps}")
        
        for _ in range(self.num_train_steps):
            # Get supervised batch
            try:
                sup_batch = next(train_iter)
            except (StopIteration, NameError):
                train_iter = iter(train_dataloader)
                sup_batch = next(train_iter)
                
            # Get unsupervised batch
            unsup_batch = None
            if unsup_dataloader and self.unsup_ratio > 0:
                try:
                    unsup_batch = next(unsup_iter)
                except (StopIteration, NameError):
                    unsup_iter = iter(unsup_dataloader)
                    unsup_batch = next(unsup_iter)
            
            # Prepare inputs
            batch = self._prepare_batch(sup_batch, unsup_batch)
            
            # Forward pass
            self.model.train()
            loss, _, loss_dict = self.model(
                **batch,
                unsup_ratio=self.unsup_ratio,
                tsa=self.tsa,
                global_step=self.global_step,
                num_train_steps=self.num_train_steps,
            )
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            self.grad_clipper()
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            
            # Log metrics
            if self.global_step % self.logging_steps == 0:
                self._log_metrics("train", loss_dict)
                
            # Save model checkpoint
            if self.logging_dir and self.global_step % self.save_steps == 0:
                output_dir = os.path.join(self.logging_dir, f"checkpoint-{self.global_step}")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
                
            self.global_step += 1
            
            # Break if we've reached the total number of training steps
            if self.global_step >= self.num_train_steps:
                break
                
        # Save final model
        if self.logging_dir:
            output_dir = os.path.join(self.logging_dir, "final")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
            
        return self.global_step
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if not self.eval_dataset:
            logging.info("No evaluation data provided")
            return {}
            
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=32,  # This should be configurable
            shuffle=False
        )
        
        # Eval loop
        self.model.eval()
        total_loss = 0
        total_examples = 0
        all_logits = []
        all_labels = []
        
        logging.info("***** Running evaluation *****")
        logging.info(f"  Num examples = {len(self.eval_dataset)}")
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Prepare inputs
                prepared_batch = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "token_type_ids": batch["token_type_ids"].to(self.device),
                    "labels": batch["labels"].to(self.device) if "labels" in batch else None,
                }
                
                # Forward pass
                loss, logits, _ = self.model(**prepared_batch)
                
                total_loss += loss.item() * len(batch["input_ids"])
                total_examples += len(batch["input_ids"])
                
                if "labels" in batch:
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(batch["labels"].cpu())
        
        # Calculate metrics
        metrics = {"eval_loss": total_loss / total_examples}
        
        if all_logits and all_labels:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            preds = torch.argmax(all_logits, dim=-1)
            acc = (preds == all_labels).float().mean().item()
            metrics["eval_accuracy"] = acc
            
        # Log metrics
        self._log_metrics("eval", metrics)
            
        return metrics
    
    def _prepare_batch(self, sup_batch, unsup_batch=None):
        """Prepare input batch for the model."""
        batch = {
            "input_ids": sup_batch["input_ids"].to(self.device),
            "attention_mask": sup_batch["attention_mask"].to(self.device),
            "token_type_ids": sup_batch["token_type_ids"].to(self.device),
            "labels": sup_batch["labels"].to(self.device) if "labels" in sup_batch else None,
        }
        
        if unsup_batch and self.unsup_ratio > 0:
            batch.update({
                "ori_input_ids": unsup_batch["ori_input_ids"].to(self.device),
                "ori_attention_mask": unsup_batch["ori_attention_mask"].to(self.device),
                "ori_token_type_ids": unsup_batch["ori_token_type_ids"].to(self.device),
                "aug_input_ids": unsup_batch["aug_input_ids"].to(self.device),
                "aug_attention_mask": unsup_batch["aug_attention_mask"].to(self.device),
                "aug_token_type_ids": unsup_batch["aug_token_type_ids"].to(self.device),
            })
            
        return batch
    
    def _log_metrics(self, prefix, metrics):
        """Log metrics to console and tensorboard."""
        log_str = f"{prefix} metrics: "
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_str += f"{key}={value:.4f}, "
            
            if self.writer:
                self.writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
                
        logging.info(log_str)


def create_uda_model(
    bert_config,
    num_labels,
    init_checkpoint=None,
):
    """Create a UDA model from config."""
    bert_model = BertModel(bert_config)
    
    if init_checkpoint:
        logging.info(f"Loading pretrained weights from {init_checkpoint}")
        state_dict = torch.load(init_checkpoint, map_location="cpu")
        # Filter out classifier weights if they exist
        model_state_dict = {k: v for k, v in state_dict.items() 
                            if not k.startswith("classifier")}
        bert_model.load_state_dict(model_state_dict, strict=False)
        
    model = UDAModel(bert_model, num_labels)
    return model 