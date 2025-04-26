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
"""Functions and classes related to multi-GPU optimization.

PyTorch implementation of the AdamWeightDecayOptimizer with multi-GPU support.
"""

import re
import torch
from torch.optim import Optimizer
import torch.distributed as dist


class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 params,
                 lr=1e-3,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta_2))
        
        defaults = dict(lr=lr, 
                        weight_decay=weight_decay_rate,
                        beta1=beta_1,
                        beta2=beta_2,
                        epsilon=epsilon)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)
        
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['epsilon'])

                # Get the parameter name for the weight decay check
                param_name = self._get_param_name(p)
                
                # Apply weight decay if needed
                update = exp_avg / denom
                if self._do_use_weight_decay(param_name):
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Update parameter
                p.data.add_(update, alpha=-group['lr'])

        return loss
    
    def _get_param_name(self, param):
        """Get the name of a parameter."""
        # In PyTorch, we need to extract the parameter name from the internal structure
        # This is a bit hacky but should work for most cases
        for name, value in self.named_parameters():
            if value is param:
                return name
        return None
    
    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.defaults['weight_decay']:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True
    
    def named_parameters(self):
        """Returns an iterator over model parameters, yielding both the
        name of the parameter and the parameter itself.
        """
        for group in self.param_groups:
            for p in group['params']:
                name = self._get_param_name(p)
                if name is not None:
                    yield name, p


class DistributedOptimizer(AdamWeightDecayOptimizer):
    """Implements Adam algorithm with weight decay and multi-GPU support."""
    
    def __init__(self,
                 params,
                 lr=1e-3,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="DistributedOptimizer"):
        """Constructs a DistributedOptimizer."""
        super(DistributedOptimizer, self).__init__(
            params,
            lr=lr,
            weight_decay_rate=weight_decay_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            exclude_from_weight_decay=exclude_from_weight_decay,
            name=name)
    
    def step(self, closure=None):
        """Performs a single optimization step with gradient synchronization.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Synchronize gradients across all devices
        self._sync_gradients()
        
        # Now perform the regular optimization step
        return super(DistributedOptimizer, self).step(None)
    
    def _sync_gradients(self):
        """Synchronize gradients across all devices."""
        world_size = dist.get_world_size()
        if world_size <= 1:
            return

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Average the gradient across all devices
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data.div_(world_size)


def get_adam_optimizer(model, learning_rate, use_distributed=False):
    """Get adam optimizer with optional distributed training support."""
    # Prepare optimizer parameters with weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() 
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    # Choose the right optimizer based on distributed setting
    optimizer_cls = DistributedOptimizer if use_distributed else AdamWeightDecayOptimizer
    
    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    
    return optimizer


def create_optimizer(model, loss, init_lr, num_train_steps, num_warmup_steps, 
                    use_distributed=False, clip_norm=1.0):
    """Creates an optimizer with learning rate schedule and gradient clipping."""
    
    # Create optimizer with appropriate distributed setting
    optimizer = get_adam_optimizer(model, init_lr, use_distributed)
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )
    
    # Return optimizer, scheduler, and gradient clipping function
    return optimizer, scheduler, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    
    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate
        num_warmup_steps (int): The number of steps for the warmup phase
        num_training_steps (int): The total number of training steps
        last_epoch (int, optional): The index of the last epoch. Default: -1
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch) 