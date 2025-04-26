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
"""PyTorch implementation of TPU utilities."""

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch_xla.core.xla_model as xm
import logging

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """A simple class to log metrics to TensorBoard."""
    
    def __init__(self, model_dir, prefix=""):
        """Initialize the logger.
        
        Args:
            model_dir: Directory to save the TensorBoard logs.
            prefix: Prefix for all metric names.
        """
        os.makedirs(model_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=model_dir)
        self.prefix = prefix
        self.step = 0
        
    def log_metrics(self, metric_dict, step=None, reduce_fn=None):
        """Log metrics to TensorBoard.
        
        Args:
            metric_dict: Dictionary of metrics to log.
            step: Global step. If None, use internal counter.
            reduce_fn: Function to reduce metric values across TPU replicas.
                If None, use the first value. Typically torch.mean.
        """
        if step is not None:
            self.step = step
            
        for name, value in metric_dict.items():
            # Make sure value is a tensor
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
                
            # Move to CPU if on TPU
            if value.device.type == 'xla':
                value = value.cpu()
                
            # Reduce across TPU replicas if needed
            if reduce_fn is not None and xm.xrt_world_size() > 1:
                value = reduce_fn(value)
            elif xm.xrt_world_size() > 1:
                # Default to taking the first value if no reduce_fn
                value = xm.mesh_reduce('scalar_reduce', value, lambda x: x[0])
                
            # Log to TensorBoard
            self.writer.add_scalar(f"{self.prefix}{name}", value, self.step)
            
        self.step += 1
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


def create_scalar_host_call(metric_dict, model_dir, prefix="", reduce_fn=None):
    """Create a function to log metrics to TensorBoard.
    
    This creates and returns a function that can be called to log metrics to TensorBoard,
    similar to TensorFlow's host_call for TPUs.
    
    Args:
        metric_dict: Dictionary of metrics to log.
        model_dir: Directory to save the TensorBoard logs.
        prefix: Prefix for all metric names.
        reduce_fn: Function to reduce metric values across TPU replicas.
            If None, use the first value. Typically torch.mean.
            
    Returns:
        A function that can be called to log metrics.
    """
    logger = TensorBoardLogger(model_dir, prefix)
    
    def host_call_fn(global_step=None):
        """Log metrics to TensorBoard.
        
        Args:
            global_step: Global step. If None, use logger's internal counter.
        """
        logger.log_metrics(metric_dict, global_step, reduce_fn)
        
    return host_call_fn


def get_tpu_resolver():
    """Get TPU resolver for PyTorch/XLA.
    
    Returns:
        TPU resolver if TPUs are available, None otherwise.
    """
    if not torch.cuda.is_available():
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_multiprocessing as xmp
            
            return xmp.MpModelWrapper()
        except ImportError:
            logger.warning("PyTorch XLA not available.")
    return None 