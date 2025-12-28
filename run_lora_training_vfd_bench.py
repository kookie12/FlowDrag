# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
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
# *************************************************************************

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import PIL
from PIL import Image

from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.lora_utils import train_lora
import time

# Paths
SD_15_PATH = "/mnt/hdd/sunjaeyoon/workspace/pretrain_SD_models/CompVis/stable-diffusion-v1-5"
INPUT_DATASET_PATH = "/mnt/hdd/sunjaeyoon/workspace/ICML2025/FlowDrag_251110/dataset/VFD_Bench_Dataset"
OUTPUT_LORA_PATH = "/mnt/hdd/sunjaeyoon/workspace/ICML2025/FlowDrag_251110/lora_data/VFD_Bench_Dataset"

def sort_folder_name(folder_name):
    """
    Key function for sorting folder names
    After splitting by '_':
    1. First part (e.g., "Pexels", "TVR", "loveu-tgve-2023")
    2. Second part as number (converted to integer)
    3. Remaining parts (compared as strings)
    4. Last number (if exists)
    """
    parts = folder_name.split('_')
    
    if not parts:
        return ('', float('inf'), [], float('inf'))
    
    first_part = parts[0]
    second_num = float('inf')
    
    if len(parts) > 1:
        try:
            second_num = int(parts[1])
        except ValueError:
            second_num = float('inf')

    remaining_parts = parts[2:] if len(parts) > 2 else []
    
    last_num = float('inf')
    if remaining_parts:
        try:
            last_num = int(remaining_parts[-1])
            remaining_parts = remaining_parts[:-1]  # Exclude the last number from remaining parts
        except ValueError:
            pass  # If the last part is not a number, keep it as is
    
    return (first_part, second_num, tuple(remaining_parts), last_num)

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.isdir(OUTPUT_LORA_PATH):
        os.makedirs(OUTPUT_LORA_PATH)

    # Get all user directories (sunjae, etc.)
    user_dirs = [d for d in os.listdir(INPUT_DATASET_PATH) if os.path.isdir(os.path.join(INPUT_DATASET_PATH, d))]
    
    # Sort user_dirs according to custom sorting rules
    user_dirs = sorted(user_dirs, key=sort_folder_name)

    # Start time measurement
    start_time = time.time()

    # Process each user's directory
    for user_dir in tqdm(user_dirs, desc="Processing Users"):
        user_path = os.path.join(INPUT_DATASET_PATH, user_dir)
        user_output_path = os.path.join(OUTPUT_LORA_PATH, user_dir)
        
        # Create user output directory if it doesn't exist
        if not os.path.isdir(user_output_path):
            os.makedirs(user_output_path)
        
        # Check for both jpg and png files
        input_image_path = None
        for ext in ['jpg', 'png', 'PNG']:
            temp_path = os.path.join(user_path, f'input.{ext}')
            if os.path.exists(temp_path):
                input_image_path = temp_path
                break
        
        if input_image_path is None:
            print(f"Warning: input.jpg or input.png not found in {user_path}")
            continue

        # Read image file
        source_image = Image.open(input_image_path)
        source_image = np.array(source_image)

        # Create output directory for this sample
        save_lora_path = user_output_path
        if not os.path.isdir(save_lora_path):
            os.makedirs(save_lora_path)

        # Train and save LoRA
        # Using a default prompt since we don't have meta_data.pkl
        prompt = "a high quality image"  # You might want to modify this or get it from somewhere
        
        train_lora(source_image, prompt,
                    model_path=SD_15_PATH,
                    vae_path="default",
                    save_lora_path=save_lora_path,
                    lora_step=80,
                    lora_lr=0.0005,
                    lora_batch_size=4,
                    lora_rank=16,
                    progress=tqdm,
                    use_gradio_progress=False)
                    # save_interval=10)
            
    # End time measurement
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print execution time
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")