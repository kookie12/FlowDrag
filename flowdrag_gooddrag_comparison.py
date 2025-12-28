import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import shutil
import pandas as pd

def calculate_distances(handle_points, target_points):
    distances = []
    for hp, tp in zip(handle_points, target_points):
        # Calculate Euclidean distance between two points
        distance = round(np.linalg.norm(np.array(hp) - np.array(tp)), 2)
        distances.append(distance)
    return distances

def txt_draw(text, v="top", h="left", target_size=[512,512]): # width, height
    
    if v == "center" and h == "center":
        fig_width, fig_height = target_size
        ratio = fig_width / fig_height
        plt.figure(dpi=300, figsize=(1,1/ratio))
        plt.text(0.5, 0.5, text, fontsize=3.5, wrap=True, verticalalignment="center", horizontalalignment="center")
    else:
        plt.figure(dpi=300, figsize=(1,1))
        plt.text(-0.1, 1.1, text, fontsize=3.5, wrap=True, verticalalignment=v, horizontalalignment=h)
        
    plt.axis('off')
    
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8) # => np.fromstring method converted strings to byte arrays, but in newer NumPy versions this can cause confusion, so it has been replaced with np.frombuffer
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    # image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # image = image.resize(target_size,Image.ANTIALIAS)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)[:,:,:3]
    
    plt.close('all')
    
    return image

# image_folder=flowdrag_sample_dir, data_folder=dataset_sample_dir,
def save_image_all(image_folder, data_folder, points, new_points, metric_df, model_name):
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    # Path to the original image with points
    data_folder = Path(data_folder)
    
    possible_extensions = ['input.jpg', 'input.png', 'input.PNG']
    input_image_path = None
    for ext in possible_extensions:
        temp_path = data_folder / ext
        if os.path.exists(str(temp_path.absolute())):
            input_image_path = temp_path
            break

    if input_image_path is None:
        raise FileNotFoundError(f"No input image found in {data_folder}. Expected one of: {possible_extensions}")
        
    input_image = Image.open(str(input_image_path))
    input_image_w_points = Image.open(str(img_folder / 'original_image_w_points.png'))
    
    gt_image = Image.open(str(data_folder / 'gt.png'))
    
    w, h = input_image.size
    
    text_image_00 = txt_draw("Input Info & Results", v="center", h="center", target_size=[w, 80])
    text_image_01 = txt_draw("Original Image", v="center", h="center", target_size=[w, 80])
    text_image_02 = txt_draw("User Drag", v="center", h="center", target_size=[w, 80])
    
    if model_name == 'gooddrag':
        text_image_03 = txt_draw("GoodDrag", v="center", h="center", target_size=[w, 80])
        text_image_04 = txt_draw("GoodDrag with points", v="center", h="center", target_size=[w, 80])
        text_image_05 = txt_draw("GT Image", v="center", h="center", target_size=[w, 80])
    elif model_name == 'flowdrag':
        text_image_03 = txt_draw("FlowDrag", v="center", h="center", target_size=[w, 80])
        text_image_04 = txt_draw("FlowDrag with points", v="center", h="center", target_size=[w, 80])
        text_image_05 = txt_draw("GT Image", v="center", h="center", target_size=[w, 80])
        
    text_image = np.concatenate([text_image_00, text_image_01, text_image_02, text_image_03, text_image_04, text_image_05], axis=1) # 가로로 합치기
    
    output_image = Image.open(str(img_folder / 'output_image.png'))
    output_image_w_points = Image.open(str(img_folder / 'output_image_w_new_points.png'))
    
    # handle_points = [point.tolist() for point in points]
    # target_points = [point.tolist() for point in new_points]
    
    original_handle_points = [points[i] for i in range(0, len(points), 2)]
    target_points = [points[i] for i in range(1, len(points), 2)]
    
    new_handle_points = [new_points[i] for i in range(0, len(new_points), 2)]
    new_target_points = [new_points[i] for i in range(1, len(new_points), 2)]
    
    # points = [[335, 164], [424, 242]], new_points = [[432, 230], [424, 242]]
    original_hp_tp_diff = calculate_distances(points[0], points[1])
    new_hp_tp_diff = calculate_distances(new_points[0], new_points[1])
    original_hp_tp_diff_total = sum(original_hp_tp_diff)
    new_hp_tp_diff_total = sum(new_hp_tp_diff)
    
    # Get metrics for the specific image folder
    metric_row = metric_df[(metric_df['filename'] == img_folder.name)]
    
    if len(metric_row) == 0:
        print(f"Warning: No metrics found for {img_folder.name} in {model_name}")
        # Set default values if metrics not found
        lpips_source = lpips_gt = psnr_source = psnr_gt = clip_sim_source = clip_sim_gt = md = 0.0
    else:
        # get metric
        lpips_source = metric_row['cur_1-lpips_source'].values[0]
        lpips_gt = metric_row['cur_1-lpips_gt'].values[0]
        
        psnr_source = metric_row['psnr_source'].values[0]
        psnr_gt = metric_row['psnr_gt'].values[0]
        
        clip_sim_source = metric_row['cur_clip_sim_source'].values[0]
        clip_sim_gt = metric_row['cur_clip_sim_gt'].values[0]
        
        md = metric_row['cur_dist'].values[0]
    
    # concat 3 images 
    # image_instruct = txt_draw(f"handle_point: {original_handle_points} \n new_handle_points: {new_handle_points} \n target_point: {target_points} \n\n hp_tp_diff: {original_hp_tp_diff} \n"
    #                             f"new_hp_tp_diff: {new_hp_tp_diff} \n diff_total: {original_hp_tp_diff_total:.2f} \n new_diff_total: {new_hp_tp_diff_total:.2f} \n time: {elapsed_time:.2f}s \n",
    #                             target_size=input_image.size)
    
    image_instruct = txt_draw(f"hp_tp_diff: {original_hp_tp_diff} \n new_hp_tp_diff: {new_hp_tp_diff} \n diff_total: {original_hp_tp_diff_total:.2f} \n new_diff_total: {new_hp_tp_diff_total:.2f} \n"
                            f"\n LPIPS source: {lpips_source:.2f} \n PSNR source: {psnr_source:.2f} \n CLIP Sim source: {clip_sim_source:.2f} \n LPIPS gt: {lpips_gt:.2f} \n PSNR gt: {psnr_gt:.2f} \n CLIP Sim gt: {clip_sim_gt:.2f} \n MD: {md:.2f} \n",
                              target_size=input_image.size)
    
    concat_image = np.concatenate([image_instruct, input_image, input_image_w_points, output_image, output_image_w_points, gt_image], axis=1) # 가로로 합치기
    total = np.concatenate([text_image, concat_image], axis=0) 
            
    # Image.fromarray(total).save(str(img_folder / 'concat_image.png'))

    # return {
    #     "handle_points": original_handle_points,
    #     "new_handle_points": new_handle_points,
    #     "target_points": target_points,
    #     "hp_tp_diff": original_hp_tp_diff,
    #     "new_hp_tp_diff": new_hp_tp_diff,
    #     "diff_total": original_hp_tp_diff_total,
    #     "new_diff_total": new_hp_tp_diff_total,
    # }

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flowdrag_root', type=str, default='VFD_Bench_result_flowdrag',
                      help='Root directory for flowdrag results')
    parser.add_argument('--gooddrag_root', type=str, default='VFD_Bench_result_gooddrag',
                      help='Root directory for gooddrag results')
    parser.add_argument('--dataset_root', type=str, default='VFD_Bench_Dataset_Final',
                      help='Root directory for dataset')
    parser.add_argument('--output_root', type=str, default='comparison_results',
                      help='Root directory for output comparison images')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_root, exist_ok=True)
    
    # Get all categories from either flowdrag or gooddrag root (they should be the same)
    categories = sorted(os.listdir(args.flowdrag_root))
    
    # load csv for metrics
    flowdrag_metric_path = os.path.join(args.flowdrag_root, "eval_similarity.csv")
    gooddrag_metric_path = os.path.join(args.gooddrag_root, "eval_similarity.csv")
    
    flowdrag_metric_df = pd.read_csv(flowdrag_metric_path)
    gooddrag_metric_df = pd.read_csv(gooddrag_metric_path)
    
    for category in tqdm(categories, desc='Processing categories'):
        if category == "Pexels_kookie" or category == "Pexels_younghwan":
            print(f"Skipping {category}")
            continue
        
        # Skip if category is a file (e.g., csv files)
        category_path = os.path.join(args.flowdrag_root, category)
        if not os.path.isdir(category_path):
            print(f"Skipping {category} as it's not a directory")
            continue
        
        flowdrag_cat_dir = os.path.join(args.flowdrag_root, category)
        gooddrag_cat_dir = os.path.join(args.gooddrag_root, category)
        dataset_cat_dir = os.path.join(args.dataset_root, category)
        output_cat_dir = os.path.join(args.output_root, category)
        
        os.makedirs(output_cat_dir, exist_ok=True)
        
        # Get all sample directories
        # sample_dirs = sorted(os.listdir(flowdrag_cat_dir), key=lambda x: int(x.split('_')[0]))
        sample_dirs = sorted(os.listdir(flowdrag_cat_dir))
        
        for sample_dir in tqdm(sample_dirs, desc=f'Processing {category}'):
            flowdrag_sample_dir = os.path.join(flowdrag_cat_dir, sample_dir)
            gooddrag_sample_dir = os.path.join(gooddrag_cat_dir, sample_dir)
            dataset_sample_dir = os.path.join(dataset_cat_dir, sample_dir)
            output_sample_dir = os.path.join(output_cat_dir, sample_dir)
            
            # os.makedirs(output_sample_dir, exist_ok=True)
            
            try:
                # Load points from dataset directory
                points_path = os.path.join(dataset_sample_dir, 'points.json')
                with open(points_path, 'r') as f:
                    points_data = json.load(f)
                    points = points_data['points']
                
                # Load new points from both flowdrag and gooddrag
                flowdrag_points_path = os.path.join(flowdrag_sample_dir, 'new_points.json')
                gooddrag_points_path = os.path.join(gooddrag_sample_dir, 'new_points.json')
                
                with open(flowdrag_points_path, 'r') as f:
                    flowdrag_points = json.load(f)['user_points']
                with open(gooddrag_points_path, 'r') as f:
                    gooddrag_points = json.load(f)['user_points']
                
                # Load elapsed time from both
                # with open(os.path.join(flowdrag_sample_dir, 'elapsed_time.txt'), 'r') as f:
                #     flowdrag_time = float(f.read().strip())
                # with open(os.path.join(gooddrag_sample_dir, 'elapsed_time.txt'), 'r') as f:
                #     gooddrag_time = float(f.read().strip())
                
                # Create comparison for flowdrag
                flowdrag_result = save_image_all(
                    flowdrag_sample_dir,
                    dataset_sample_dir,
                    points,
                    flowdrag_points,
                    flowdrag_metric_df,
                    'flowdrag'
                )
                
                # Create comparison for gooddrag
                gooddrag_result = save_image_all(
                    gooddrag_sample_dir,
                    dataset_sample_dir,
                    points,
                    gooddrag_points,
                    gooddrag_metric_df,
                    'gooddrag'
                )
                
                # Concatenate vertically
                final_image = np.concatenate([flowdrag_result, gooddrag_result], axis=0)
                
                # Save the final comparison image
                # output_path = os.path.join(output_sample_dir, 'comparison.png')
                output_path = f'{output_sample_dir}.png'
                Image.fromarray(final_image).save(output_path)
                
            except Exception as e:
                print(f"Error processing {sample_dir} in {category}: {str(e)}")
                continue
    