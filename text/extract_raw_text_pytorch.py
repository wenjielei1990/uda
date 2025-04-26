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
"""Extract raw text for back translation. PyTorch implementation.
"""

import os
import argparse
import logging
from tqdm import tqdm

from utils import raw_data_utils


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def dump_raw_examples(examples, output_dir, separate_doc_by_newline):
    """Dump raw examples to text files."""
    logger.info("Dumping raw examples")
    
    text_path = os.path.join(output_dir, "text.txt")
    label_path = os.path.join(output_dir, "label.txt")
    
    with open(text_path, "w") as text_ouf:
        with open(label_path, "w") as label_ouf:
            for example in tqdm(examples, desc="Writing examples"):
                text_a = example.text_a
                text_b = example.text_b
                label = example.label
                
                text_ouf.write(text_a + "\n")
                if text_b is not None:
                    text_ouf.write(text_b + "\n")
                if separate_doc_by_newline:
                    text_ouf.write("\n")
                    
                label_ouf.write(label + "\n")
                
    logger.info(f"Finished dumping raw examples to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract raw text for back translation")
    
    parser.add_argument(
        "--separate_doc_by_newline", 
        action="store_true",
        help="Whether to separate documents by newline"
    )
    
    parser.add_argument(
        "--output_data_dir",
        type=str,
        required=True,
        help="Directory to save output data"
    )
    
    parser.add_argument(
        "--sub_set",
        type=str,
        default="unsup_in",
        help="Which subset to process"
    )
    
    parser.add_argument(
        "--task_name",
        type=str,
        default="IMDB",
        help="Name of the task"
    )
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory containing raw data"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    processor = raw_data_utils.get_processor(args.task_name)
    logger.info("Loading examples")
    
    # Create output directory
    output_dir = os.path.join(args.output_data_dir, args.sub_set)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load examples based on subset
    if args.sub_set == "train":
        examples = processor.get_train_examples(args.raw_data_dir)
    elif args.sub_set.startswith("unsup"):
        examples = processor.get_unsup_examples(args.raw_data_dir, args.sub_set)
    else:
        raise ValueError(f"Unsupported subset: {args.sub_set}")
        
    logger.info("Finished loading examples")
    logger.info(f"Number of examples: {len(examples)}")
    
    # Dump examples to files
    dump_raw_examples(examples, output_dir, args.separate_doc_by_newline)


if __name__ == '__main__':
    main() 