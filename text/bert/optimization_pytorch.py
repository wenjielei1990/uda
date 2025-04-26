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
"""Functions and classes related to optimization (weight updates).

PyTorch implementation of the optimization code.
"""

import re
import math
import torch
from torch.optim import Optimizer


class AdamWeightDecayOptimizer(Optimizer):
    """Implements Adam algorithm with weight decay correctly.
    
    This is different from the standard PyTorch AdamW implementation in that
    it implements weight decay correctly (following the original Adam paper).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, exclude_from_weight_decay=None):
        """Constructs a AdamWeightDecayOptimizer."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                
                update = exp_avg / denom
                
                # Apply weight decay only if parameter name matches requirements
                param_name = self._get_param_name(p)
                if self._do_use_weight_decay(param_name, group['weight_decay']):
                    update.add_(p.data, alpha=group['weight_decay'])
                
                p.data.add_(update, alpha=-step_size)

        return loss

    def _get_param_name(self, param):
        """Get the name of a parameter."""
        # In PyTorch, we need to extract the parameter name from the internal structure
        # This is a bit hacky but should work for most cases
        for name, value in self.named_parameters():
            if value is param:
                return name
        return None

    def _do_use_weight_decay(self, param_name, weight_decay_rate):
        """Whether to use L2 weight decay for `param_name`."""
        if not weight_decay_rate:
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


def get_adam_optimizer(model, learning_rate):
    """Get adam optimizer."""
    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    
    return optimizer


def create_optimizer(model, loss, init_lr, num_train_steps, num_warmup_steps,
                      clip_norm=1.0):
    """Creates an optimizer with learning rate schedule."""
    
    # Create the optimizer
    optimizer = get_adam_optimizer(model, init_lr)
    
    # Create learning rate scheduler
    scheduler = LinearWarmupScheduler(
        optimizer, 
        warmup_steps=num_warmup_steps,
        total_steps=num_train_steps
    )
    
    # Define gradient clipping function
    def clip_gradients(model, clip_norm):
        """Clip gradients of the model."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    
    # Return optimizer, scheduler and gradient clipping function
    return optimizer, scheduler, clip_gradients


class LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements linear warmup and linear decay for learning rate schedule."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        """Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Get learning rate based on current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            scale = float(step) / max(1.0, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Linear decay
            scale = max(0.0, float(self.total_steps - step) / 
                        max(1.0, self.total_steps - self.warmup_steps))
            return [base_lr * scale for base_lr in self.base_lrs] 