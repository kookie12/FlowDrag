# *************************************************************************
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
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
from utils.ui_utils import run_gooddrag, run_flowdrag, train_lora_interface, show_cur_points, create_video, save_image_all
from tqdm import tqdm
import time
import pandas as pd

def benchmark_dataset(dataset_folder):
    
    global_start_time = time.time()
    
    dataset_path = Path(dataset_folder)
    subfolders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name != '.ipynb_checkpoints']
    subfolders = sorted(subfolders, key=lambda x: x.name)
    
    df = pd.DataFrame({
        'sample_name': [],
        'handle_points': [], 
        'new_handle_points': [],   
        'target_points': [],
        'hp_tp_diff': [],
        'new_hp_tp_diff': [],
        'diff_total': [],
        'new_diff_total': [],
        'time': [],   
        'time_hms': [],   
    })
    df_total = df.copy()
    df_total_path = result_dir
    
    for subfolder in tqdm(subfolders):        
        print(f'Benchmarking {subfolder.name}')
        
        results = bench_one_image(subfolder)
        
        if results is None:
            print(f"Skipping {subfolder.name} as results already exist in {results['result_dir']}")
            continue
        
        df.loc[len(df)] = [subfolder.name, str(results['handle_points']), str(results['new_handle_points']), str(results['target_points']), str(results['hp_tp_diff']), \
            str(results['new_handle_points']), str(results['diff_total']), str(results['new_diff_total']), results['elapsed_time'], results['time_hms']]
        
        df_total.loc[len(df)] = [subfolder.name, str(results['handle_points']), str(results['new_handle_points']), str(results['target_points']), str(results['hp_tp_diff']), \
            str(results['new_handle_points']), str(results['diff_total']), str(results['new_diff_total']), results['elapsed_time'], results['time_hms']]
        
        df.to_csv(os.path.join(results['result_dir'], 'result.csv'), index=True)
        df_total.to_csv(os.path.join(df_total_path, 'result_total.csv'), index=True)
            
    global_end_time = time.time()
    _total_time = global_end_time - global_start_time
    hours, remainder = divmod(_total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    print(f'All benchmarking finished!')
    print(f'Total time: {total_time}')
    

def load_data(folder):
    """Load the original image, mask, and points from the specified folder."""
    folder_path = Path(folder)

    # Load original image
    original_image_path = folder_path / 'input.png'
    
    if not os.path.exists(str(original_image_path.absolute())):
        raise FileNotFoundError(f"No input.png found in {folder_path}")
        
    original_image = Image.open(original_image_path)
    original_image = np.array(original_image)

    # Load mask
    mask_path = folder_path / 'mask.png'
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Load points
    if model_name == 'gooddrag':
        points_path = folder_path / 'points.json'
    
    elif model_name == 'flowdrag':
        points_path = folder_path / 'points_flowdrag.json'
    
    with open(points_path, 'r') as f:
        points_data = json.load(f)
        points = points_data['points']

    # image_points_path = folder_path / 'image_with_points.jpg'
    # image_with_points = Image.open(image_points_path)
    # image_with_points = np.array(image_with_points)

    return original_image, mask, points # , image_with_points


def bench_one_image(folder): # folder = PosixPath('dataset/VFD_Bench_Dataset/Pexels_11488418_deer_haed_move_0')
    """ 
    Test the saved data by running the drag model.

    Args:
      folder: The folder where the original image, mask, and points are saved.
    """
    # original_image, mask, points, image_with_points = load_data(folder)
    original_image, mask, points = load_data(folder)
    model_path = '/mnt/hdd/sunjaeyoon/workspace/pretrain_SD_models/CompVis/stable-diffusion-v1-5'
    VAE_PATH = 'default'

    lora_path = str(folder.absolute()).replace(folder.parts[-3], lora_data)

    if os.path.exists(lora_path):
        print(f'Lora data exists. Skip training Lora.')

    else: 
        print(f'Training Lora.')
        train_lora_interface(original_image=original_image, prompt='', model_path=model_path,
                            vae_path=VAE_PATH, #'stabilityai/sd-vae-ft-mse',
                            lora_path=lora_path, lora_step=70, lora_lr=0.0005, lora_batch_size=4, lora_rank=16,
                            use_gradio_progress=False)
        print(f'Training Lora Done! Begin dragging.')

    return_intermediate_images = False

    sample_result_dir = f'{result_dir}/{Path(folder).parts[-1]}'
    # if os.path.exists(sample_result_dir):
    #     print(f"Skipping {Path(folder).parts[-1]} as results already exist in {sample_result_dir}")
    #     return None
    os.makedirs(sample_result_dir, exist_ok=True)

    start_time = time.time()

    # Select function based on model_name
    if model_name == 'flowdrag':
        drag_function = run_flowdrag
    elif model_name == 'gooddrag':
        drag_function = run_gooddrag
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Must be 'flowdrag' or 'gooddrag'")

    output_image, new_points = drag_function(
        source_image=original_image,
        mask=mask,
        prompt='',
        points=points,
        inversion_strength=0.75,
        lam=0.1,
        latent_lr=0.02,
        model_path=model_path,
        vae_path=VAE_PATH, #'stabilityai/sd-vae-ft-mse',
        lora_path=lora_path,
        drag_end_step=7,
        track_per_step=10,
        save_intermedia=False,
        compare_mode=False,
        r1=4,
        r2=12,
        d=4,
        max_drag_per_track=3,
        drag_loss_threshold=0,
        once_drag=False,
        max_track_no_change=5,
        return_intermediate_images=return_intermediate_images,
        result_save_path=sample_result_dir
    )

    print(f'Drag finished!')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_image_path = os.path.join(sample_result_dir, 'output_image.png')
    Image.fromarray(output_image).save(output_image_path)
    # cv2.imwrite(output_image_path, output_image)

    # points = [210, 414], [212, 416], [222, 352], [221, 353], [168, 298], [168, 299]]
    # 0, 2, 4 index => handle point
    # 1, 3, 5 index => target point

    original_img_with_points = show_cur_points(np.ascontiguousarray(original_image), points, bgr=True) 
    original_img_with_points_path = os.path.join(sample_result_dir, 'original_image_w_points.png')
    Image.fromarray(original_img_with_points).save(original_img_with_points_path)
    # cv2.imwrite(original_points_image_path, img_with_original_points)

    # output image는 이상하게 BGR이기 때문에, Image.fromarray().save로 저장하지 않고 cv2로 저장하겠다
    # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    output_img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    output_img_with_new_points_path = os.path.join(sample_result_dir, 'output_image_w_new_points.png')
    Image.fromarray(output_img_with_new_points).save(output_img_with_new_points_path)
    
    # cv2.imwrite(output_img_with_new_points_path, output_img_with_new_points)
    # cv2.imwrite(new_points_image_path, img_with_new_points)

    points_path = os.path.join(sample_result_dir, f'new_points.json')
    # with open(points_path, 'w') as f:
    #     json.dump({'points': new_points}, f)
    
    with open(points_path, 'w') as f:
        json.dump({'user_points': points, 'new_points': new_points}, f)

    # if return_intermediate_images:
    #     create_video(sample_result_dir, folder)

    results = save_image_all(sample_result_dir, folder, points, new_points, elapsed_time, model_name)
    results['elapsed_time'] = elapsed_time
    results['time_hms'] = total_time
    results['result_dir'] = sample_result_dir

    return results

def main(dataset_folder):
    benchmark_dataset(dataset_folder)

if __name__ == '__main__':    
    global result_dir, lora_data, model_name
    dataset = 'dataset/VFD_Bench_Dataset'
    result_dir = './VFD_Bench_result_flowdrag'
    # result_dir = './VFD_Bench_result_gooddrag'
    lora_data = 'lora_data' 
    model_name = 'flowdrag'
    # model_name = 'gooddrag'
    main(dataset)
