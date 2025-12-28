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
import shutil
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

# from diffusers import DDIMScheduler, AutoencoderKL
from pipeline import FlowDrag

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .lora_utils import train_lora
import matplotlib.pyplot as plt
import seaborn as sns

from .sampling_utils import plot_vectors, grid_based_sampling, stratified_importance_sampling

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# -------------- general UI functionality --------------
def clear_all(length=512):
    return gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           [], None, None


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


# user click the image to get points, and show the points on the image
def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def show_cur_points(_img,
                    sel_pix,
                    bgr=False):
    img = _img.copy() 
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            red = (255, 0, 0) if not bgr else (0, 0, 255)
            cv2.circle(img, tuple(point), 8, red, -1) 
        else:
            # draw a blue circle at the handle point
            blue = (0, 0, 255) if not bgr else (255, 0, 0)
            cv2.circle(img, tuple(point), 8, blue, -1) 
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


# clear all handle/target points
def undo_points(original_image,
                mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []


def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files and directories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file or a directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents


def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress(),
                         use_gradio_progress=True):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path)

    clear_folder(lora_path)

    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
        use_gradio_progress)
    return "Training LoRA Done!"


def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def save_images_with_pillow(images, base_filename='image'):
    for index, img in enumerate(images):
        # Convert array to Image object and save
        img_pil = Image.fromarray(img)
        folder_path = f'./save'
        filename = os.path.join(folder_path, "{}_{}.png".format(base_filename, index))
        img_pil.save(filename)
        print(f"Saved: {filename}")


def get_original_points(handle_points: List[torch.Tensor],
                        full_h: int,
                        full_w: int,
                        sup_res_w,
                        sup_res_h,
                        ) -> List[torch.Tensor]:
    """
    Convert local handle points and target points back to their original UI coordinates.

    Args:
        sup_res_h: Half original height of the UI canvas.
        sup_res_w: Half original width of the UI canvas.
        handle_points: List of handle points in local coordinates.
        full_h: Original height of the UI canvas.
        full_w: Original width of the UI canvas.

    Returns:
        original_handle_points: List of handle points in original UI coordinates.
    """
    original_handle_points = []

    for cur_point in handle_points:
        original_point = torch.round(
            torch.tensor([cur_point[1] * full_w / sup_res_w, cur_point[0] * full_h / sup_res_h]))
        original_handle_points.append(original_point)

    return original_handle_points


def save_image_mask_points(mask, points, image_with_points, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(output_dir, f"mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    # Save points
    points_path = os.path.join(output_dir, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f)

    image_with_points_path = os.path.join(output_dir, "image_with_points.jpg")
    Image.fromarray(image_with_points).save(image_with_points_path)

    return image_with_points


def save_drag_result(output_image, new_points, result_path):
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    result_dir = f'{result_path}'
    os.makedirs(result_dir, exist_ok=True)
    output_image_path = os.path.join(result_dir, 'output_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    # concat the image with new points and the output image
    # concat_image = np.concatenate((output_image, img_with_new_points), axis=1)
    # concat_image_path = os.path.join(result_dir, 'concat_image.png')
    # cv2.imwrite(concat_image_path, concat_image)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)

#mask, selected_points, input_image, output_image_gooddrag, new_points_gooddrag, output_flowdrag_image, new_points_flowdrag, result_path
def kookie_save_result(mask, selected_points, input_image, output_image_gooddrag, new_points_gooddrag, output_image_flowdrag, new_points_flowdrag, result_path):
    # Ensure the result directory exists
    os.makedirs(result_path, exist_ok=True)

    # Save mask
    mask_path = os.path.join(result_path, "mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    # Save the original image with selected points
    image_with_points = show_cur_points(np.ascontiguousarray(input_image), selected_points, bgr=False)
    image_with_points = cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR)
    image_with_points_path = os.path.join(result_path, "image_with_points.jpg")
    Image.fromarray(image_with_points).save(image_with_points_path)
    
    # Process GoodDrag results
    output_image_gooddrag_bgr = cv2.cvtColor(output_image_gooddrag, cv2.COLOR_RGB2BGR)
    output_image_gooddrag_path = os.path.join(result_path, 'output_image_gooddrag.png')
    cv2.imwrite(output_image_gooddrag_path, output_image_gooddrag_bgr)

    img_with_new_points_gooddrag = show_cur_points(np.ascontiguousarray(output_image_gooddrag), new_points_gooddrag, bgr=True)
    img_with_new_points_gooddrag = cv2.cvtColor(img_with_new_points_gooddrag, cv2.COLOR_RGB2BGR)
    new_points_gooddrag_image_path = os.path.join(result_path, 'image_with_new_points_gooddrag.png')
    cv2.imwrite(new_points_gooddrag_image_path, img_with_new_points_gooddrag)

    # concat_image_gooddrag = np.concatenate((image_with_points, img_with_new_points_gooddrag, output_image_gooddrag_bgr), axis=1)
    # concat_image_gooddrag_path = os.path.join(result_path, 'concat_image_gooddrag.png')
    # cv2.imwrite(concat_image_gooddrag_path, concat_image_gooddrag)

    # Process FlowDrag results
    output_image_flowdrag_bgr = cv2.cvtColor(output_image_flowdrag, cv2.COLOR_RGB2BGR)
    output_image_flowdrag_path = os.path.join(result_path, 'output_image_flowdrag.png')
    cv2.imwrite(output_image_flowdrag_path, output_image_flowdrag_bgr)

    img_with_new_points_flowdrag = show_cur_points(np.ascontiguousarray(output_image_flowdrag), new_points_flowdrag, bgr=True)
    img_with_new_points_flowdrag = cv2.cvtColor(img_with_new_points_flowdrag, cv2.COLOR_RGB2BGR)
    new_points_flowdrag_image_path = os.path.join(result_path, 'image_with_new_points_flowdrag.png')
    cv2.imwrite(new_points_flowdrag_image_path, img_with_new_points_flowdrag)

    # Create a white space on the left for the text
    height = max(image_with_points.shape[0], img_with_new_points_gooddrag.shape[0], output_image_gooddrag_bgr.shape[0], img_with_new_points_flowdrag.shape[0], output_image_flowdrag_bgr.shape[0])
    width = 300  # width of the white space
    white_space = np.ones((height, width, 3), dtype=np.uint8) * 255  # Create a white image

    # Add text to the white space
    # cv2.putText(white_space, "Results \n ", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # Add text to the white space
    cv2.putText(white_space, "Results", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(white_space, "Original Image + Points", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(white_space, "GoodDrag + New Points", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(white_space, "GoodDrag", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(white_space, "FlowDrag + New Points", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(white_space, "FlowDrag", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    concat_image_flowdrag = np.concatenate((white_space, image_with_points, img_with_new_points_gooddrag, output_image_gooddrag_bgr, img_with_new_points_flowdrag, output_image_flowdrag_bgr), axis=1)
    concat_image_flowdrag_path = os.path.join(result_path, 'concat_image_flowdrag.png')
    cv2.imwrite(concat_image_flowdrag_path, concat_image_flowdrag)

    # Save points and new points
    points_path = os.path.join(result_path, "points.json")
    data = {
        "selected_points": selected_points,
        "new_points_gooddrag": new_points_gooddrag,
        "new_points_flowdrag": new_points_flowdrag
    }
    with open(points_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Results saved successfully in {result_path}")
    
    # result_dir = f'{result_path}'
    # os.makedirs(result_dir, exist_ok=True)
    # # Save mask
    # mask_path = os.path.join(result_path, f"mask.png")
    # Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    # image_with_points_path = os.path.join(result_path, "image_with_points.jpg")
    # Image.fromarray(image_with_points).save(image_with_points_path)
    # image_with_points = cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR)
    
    # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    # output_image_path = os.path.join(result_dir, 'output_image.png')
    # cv2.imwrite(output_image_path, output_image)

    # img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    # new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    # cv2.imwrite(new_points_image_path, img_with_new_points)

    # # concat the image with new points and the output image
    # concat_image = np.concatenate((image_with_points, img_with_new_points, output_image), axis=1)
    # concat_image_path = os.path.join(result_dir, 'concat_image.png')
    # cv2.imwrite(concat_image_path, concat_image)

    # # Save points and new points
    # points_path = os.path.join(result_path, f"points.json")
    # data = {
    #     "points": points,
    #     "new_points": new_points
    # }
    # with open(points_path, 'w') as f:
    #     json.dump(data, f, indent=4)


def load_and_visualize_vector_field(npy_path, uploaded_image=None, selected_points=None):
    # Load the vector field from the .npy file
    try:
        vector_field = np.load(npy_path)
    except Exception as e:
        return f"Error loading npy file: {str(e)}", None

    # Create X and Y coordinate grids
    height, width, _ = vector_field.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Use uploaded image as background if provided
    background_image = None
    if uploaded_image is not None:
        try:
            # uploaded_image is already a numpy array from Gradio
            background_image = uploaded_image.copy()
            print(f"Original uploaded image shape: {background_image.shape}")
            print(f"Vector field dimensions: {height}x{width}")

            # Resize to match vector field dimensions
            if background_image.shape[:2] != (height, width):
                from PIL import Image
                pil_img = Image.fromarray(background_image)
                pil_img = pil_img.resize((width, height))
                background_image = np.array(pil_img)
                print(f"Resized background image to: {background_image.shape}")

            # Ensure image is in correct format (0-255 uint8)
            if background_image.dtype != np.uint8:
                if background_image.max() <= 1.0:
                    background_image = (background_image * 255).astype(np.uint8)
                else:
                    background_image = background_image.astype(np.uint8)

            print(f"Final background image: shape={background_image.shape}, dtype={background_image.dtype}, range=[{background_image.min()}, {background_image.max()}]")
        except Exception as e:
            print(f"Failed to use uploaded image as background: {e}")
            background_image = None
    else:
        print("No uploaded image provided for background")

    # Extract U and V components from the vector field
    U = vector_field[:, :, 0]
    V = vector_field[:, :, 1]

    # Use smaller step size for denser visualization (following flowdrag_integrated.py)
    step = 5  # Changed from 10 to 5 for better density

    # Calculate magnitude and filter significant vectors
    magnitude = np.sqrt(U**2 + V**2)
    threshold = 0.1  # Only show vectors with magnitude > 0.1

    # Calculate adaptive scale based on vector magnitudes
    significant_magnitudes = magnitude[magnitude > threshold]
    if len(significant_magnitudes) > 0:
        # Use 75th percentile of significant magnitudes to determine scale
        scale_factor = np.percentile(significant_magnitudes, 75)
        # Adaptive scaling: larger vectors get more reasonable display size
        adaptive_scale = max(1.0, min(10.0, 20.0 / scale_factor)) if scale_factor > 0 else 3.0
    else:
        adaptive_scale = 3.0

    print(f"Vector field stats: max_magnitude={magnitude.max():.3f}, adaptive_scale={adaptive_scale:.3f}")

    figsize = (8, 8)
    plt.figure(figsize=figsize)

    print(f"Background image check: background_image is {'None' if background_image is None else 'available'}")

    if background_image is not None:
        print(f"Using original background image: shape={background_image.shape}")
        # Use only the original image as background
        plt.imshow(background_image, extent=(0, width, height, 0), origin='upper', alpha=0.7)
        print(f"Background displayed: {background_image.shape}, min={background_image.min()}, max={background_image.max()}")
    else:
        print("No background image available - using white background")
        # Use simple white background if no image is provided
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        plt.imshow(white_bg, extent=(0, width, height, 0), origin='upper', alpha=0.3)

    # Plot the vectors using original .npy coordinate system
    # Filter vectors by magnitude for cleaner visualization
    mask = magnitude[::step, ::step] > threshold
    X_filtered = X[::step, ::step][mask]
    Y_filtered = Y[::step, ::step][mask]
    U_filtered = U[::step, ::step][mask]
    V_filtered = V[::step, ::step][mask]

    # Make vectors more visible with simpler settings
    plt.quiver(
        X_filtered, Y_filtered,
        U_filtered, V_filtered,
        angles='xy', scale_units='xy', scale=1.0/adaptive_scale,
        color='red', width=0.006, linewidth=1.0
    )
    print(f"Plotted {len(X_filtered)} vectors")

    # Plot selected points as red arrows (same as other vectors)
    if selected_points is not None and len(selected_points) > 0:
        selected_X, selected_Y, selected_U, selected_V = [], [], [], []
        for point in selected_points:
            x, y = point[0], point[1]
            if 0 <= y < height and 0 <= x < width:
                selected_X.append(x)
                selected_Y.append(y)
                selected_U.append(U[y, x])
                selected_V.append(V[y, x])

        if len(selected_X) > 0:
            plt.quiver(
                selected_X, selected_Y, selected_U, selected_V,
                angles='xy', scale_units='xy', scale=1.0/adaptive_scale,
                color='red', width=0.008, linewidth=1.5, zorder=10
            )
            print(f"Plotted {len(selected_X)} selected points as red arrows")

    # Set labels and title
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title('2D Vector Field of Mesh Deformation', fontsize=16)

    # Use original .npy coordinate system (image coordinates: Y-axis flipped)
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Original .npy coordinate system: top-left origin
    plt.grid(True, alpha=0.2)  # Simple grid
    
    # Save the plot to a temporary file
    plot_path = npy_path.replace(".npy", ".png") # "temp_vector_field.png"
    plt.savefig(plot_path)
    plt.close()
    
    return "Vector field loaded and visualized successfully.", plot_path

def normal_sampling_vector_field(npy_path, sampling_num, grid_size=10, uploaded_image=None, selected_points=None):
    # Load the vector field from the .npy file
    try:
        vector_field = np.load(npy_path)
    except Exception as e:
        return f"Error loading npy file: {str(e)}", None

    sampling_num = int(sampling_num)
    grid_size = int(grid_size)

    # Create X and Y coordinate grids
    height, width, _ = vector_field.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Use uploaded image as background if provided
    background_image = None
    if uploaded_image is not None:
        try:
            # uploaded_image is already a numpy array from Gradio
            background_image = uploaded_image.copy()
            # Resize to match vector field dimensions
            if background_image.shape[:2] != (height, width):
                from PIL import Image
                pil_img = Image.fromarray(background_image)
                pil_img = pil_img.resize((width, height))
                background_image = np.array(pil_img)
            print(f"Using uploaded image as background for normal sampling (size: {background_image.shape})")
        except Exception as e:
            print(f"Failed to use uploaded image as background: {e}")
            background_image = None

    # Extract U and V components from the vector field
    U = vector_field[:, :, 0]
    V = vector_field[:, :, 1]

    # Calculate the magnitude of each vector
    magnitudes = np.sqrt(U**2 + V**2)

    # Apply clipping based on a magnitude threshold
    min_magnitude_threshold = 15
    clipping_mask = magnitudes > min_magnitude_threshold
    X_clipped = X[clipping_mask]
    Y_clipped = Y[clipping_mask]
    U_clipped = U[clipping_mask]
    V_clipped = V[clipping_mask]
    
    # Perform grid-bㄴㅇㅁㅇㄴㄴㅁㅇsed normal sampling
    X_sampled, Y_sampled, U_sampled, V_sampled = grid_based_sampling(
        X_clipped, Y_clipped, U_clipped, V_clipped, width, height, grid_size=grid_size, sampling_num=sampling_num
    )

    # Add selected_points to sampled vectors (if provided)
    # NOTE: selected_points are in 512x512 coordinate system, need to scale to actual image size
    # selected_points format: [[handle_x, handle_y], [target_x, target_y], [handle_x2, handle_y2], [target_x2, target_y2], ...]
    user_points_for_display = selected_points  # Pass all for visualization
    num_user_points_added = 0

    if selected_points is not None and len(selected_points) >= 2:
        # Calculate scale factors from 512x512 to actual image size
        scale_x = width / 512.0
        scale_y = height / 512.0
        print(f"Normal sampling: Scaling {len(selected_points)//2} user points from 512x512 to {width}x{height}")

        for i in range(0, len(selected_points), 2):
            if i + 1 < len(selected_points):
                handle = selected_points[i]
                target = selected_points[i + 1]

                # Scale from 512x512 to actual image size
                x_512, y_512 = handle[0], handle[1]
                target_x_512, target_y_512 = target[0], target[1]

                x = int(x_512 * scale_x)
                y = int(y_512 * scale_y)
                target_x = int(target_x_512 * scale_x)
                target_y = int(target_y_512 * scale_y)

                # Calculate vector in actual image coordinates
                u = target_x - x
                v = target_y - y

                if 0 <= y < height and 0 <= x < width:
                    X_sampled = np.append(X_sampled, x)
                    Y_sampled = np.append(Y_sampled, y)
                    U_sampled = np.append(U_sampled, u)
                    V_sampled = np.append(V_sampled, v)
                    num_user_points_added += 1
        print(f"Added {num_user_points_added} user point pairs to sampling")

    # Dynamically determine figure size => 생각해보니까 이 figsize는 안쓰이고 plot_vectors 내부에서 사용되는 size로 되는듯?
    if width == height:
        figsize = (8, 8)
    else:
        aspect_ratio = width / height
        figsize = (8 * aspect_ratio, 8) if width > height else (8, 8 / aspect_ratio)

    # figsize = (8, 8)

    # magnitude average
    magnitude_average = np.mean(np.sqrt(U_sampled**2 + V_sampled**2))
    print(f"Magnitude average: {magnitude_average:.3f}")

    # Plot the clipped vector field
    title= 'Grid-Based Normal Sampling'
    plt.close('all')  # Close all previous figures to prevent accumulation
    fig = plt.figure(figsize=figsize)
    plt.clf()  # Clear current figure
    # Don't pass selected_points to avoid blue arrow accumulation
    # User points are already included in X_sampled and will be shown in red
    plot_vectors(
        X_sampled, Y_sampled, U_sampled, V_sampled,
        title, width, height, arrow_scale=0.3, arrow_width=0.01, grid_size=grid_size, background_image=background_image, selected_points=None
    )

    # Save to file with timestamp to avoid caching issues
    import time
    timestamp = int(time.time() * 1000)
    plot_path = npy_path.replace(".npy", f"_normal_sampled_{timestamp}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=100)

    # Also convert to numpy array for direct return to Gradio
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    plot_image = np.asarray(buf)[:, :, :3]  # Extract RGB channels only

    plt.close('all')  # Close all figures after saving

    # Save sampled points to JSON (convert back to 512x512 coordinates)
    sampled_points_512 = []
    scale_x_inv = 512.0 / width
    scale_y_inv = 512.0 / height

    for i in range(len(X_sampled)):
        x_512 = int(X_sampled[i] * scale_x_inv)
        y_512 = int(Y_sampled[i] * scale_y_inv)
        # Store as handle-target pairs
        target_x_512 = int((X_sampled[i] + U_sampled[i]) * scale_x_inv)
        target_y_512 = int((Y_sampled[i] + V_sampled[i]) * scale_y_inv)

        sampled_points_512.append([x_512, y_512])
        sampled_points_512.append([target_x_512, target_y_512])

    # Save to JSON
    directory = os.path.dirname(npy_path)
    json_path = os.path.join(directory, "points_flowdrag.json")
    with open(json_path, 'w') as f:
        json.dump({"sampled_points": sampled_points_512}, f)
    print(f"Saved {len(sampled_points_512)//2} sampled point pairs to {json_path}")

    # Return both image (as numpy array) and sampled points (in 512x512 coordinates)
    return plot_image, sampled_points_512
 
    
def importance_sampling_vector_field(npy_path, sampling_num, grid_size=10, clip_threshold=5.0, uploaded_image=None, selected_points=None):
    # Load the vector field from the .npy file
    try:
        vector_field = np.load(npy_path)
    except Exception as e:
        return f"Error loading npy file: {str(e)}", None

    sampling_num = int(sampling_num)
    grid_size = int(grid_size)

    # Create X and Y coordinate grids
    height, width, _ = vector_field.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Use uploaded image as background if provided
    background_image = None
    if uploaded_image is not None:
        try:
            # uploaded_image is already a numpy array from Gradio
            background_image = uploaded_image.copy()
            # Resize to match vector field dimensions
            if background_image.shape[:2] != (height, width):
                from PIL import Image
                pil_img = Image.fromarray(background_image)
                pil_img = pil_img.resize((width, height))
                background_image = np.array(pil_img)
            print(f"Using uploaded image as background for importance sampling (size: {background_image.shape})")
        except Exception as e:
            print(f"Failed to use uploaded image as background: {e}")
            background_image = None

    # Extract U and V components from the vector field
    U = vector_field[:, :, 0]
    V = vector_field[:, :, 1]

    # Calculate the magnitude of each vector
    magnitudes = np.sqrt(U**2 + V**2)

    # Apply clipping based on a magnitude threshold
    min_magnitude_threshold = 15
    clipping_mask = magnitudes > min_magnitude_threshold
    X_clipped = X[clipping_mask]
    Y_clipped = Y[clipping_mask]
    U_clipped = U[clipping_mask]
    V_clipped = V[clipping_mask]
    magnitudes_clipped = magnitudes[clipping_mask]

    # Perform stratified importance sampling
    X_sampled, Y_sampled, U_sampled, V_sampled = stratified_importance_sampling(
        X_clipped, Y_clipped, U_clipped, V_clipped, magnitudes_clipped, width, height,
        grid_size=grid_size, sampling_num=sampling_num, clip_threshold=clip_threshold
    )

    # Add selected_points to sampled vectors (if provided)
    # NOTE: selected_points are in 512x512 coordinate system, need to scale to actual image size
    # selected_points format: [[handle_x, handle_y], [target_x, target_y], [handle_x2, handle_y2], [target_x2, target_y2], ...]
    user_points_for_display = selected_points  # Pass all for visualization
    num_user_points_added = 0

    if selected_points is not None and len(selected_points) >= 2:
        # Calculate scale factors from 512x512 to actual image size
        scale_x = width / 512.0
        scale_y = height / 512.0
        print(f"Importance sampling: Scaling {len(selected_points)//2} user points from 512x512 to {width}x{height}")

        for i in range(0, len(selected_points), 2):
            if i + 1 < len(selected_points):
                handle = selected_points[i]
                target = selected_points[i + 1]

                # Scale from 512x512 to actual image size
                x_512, y_512 = handle[0], handle[1]
                target_x_512, target_y_512 = target[0], target[1]

                x = int(x_512 * scale_x)
                y = int(y_512 * scale_y)
                target_x = int(target_x_512 * scale_x)
                target_y = int(target_y_512 * scale_y)

                # Calculate vector in actual image coordinates
                u = target_x - x
                v = target_y - y

                if 0 <= y < height and 0 <= x < width:
                    X_sampled = np.append(X_sampled, x)
                    Y_sampled = np.append(Y_sampled, y)
                    U_sampled = np.append(U_sampled, u)
                    V_sampled = np.append(V_sampled, v)
                    num_user_points_added += 1
        print(f"Added {num_user_points_added} user point pairs to sampling")

    # magnitude average
    magnitude_average = np.mean(np.sqrt(U_sampled**2 + V_sampled**2))
    print(f"Magnitude average: {magnitude_average:.3f}")

    # Dynamically determine figure size
    max_size = 8  # Maximum size for any dimension
    if width == height:
        figsize = (max_size, max_size)  # Square figure
    else:
        aspect_ratio = width / height
        if aspect_ratio > 1:  # Width is greater than height
            figsize = (max_size, max_size / aspect_ratio)
        else:  # Height is greater than width
            figsize = (max_size * aspect_ratio, max_size)

    # figsize = (8, 8)

    # Plot the clipped vector field
    title = 'Importance Sampling'
    plt.close('all')  # Close all previous figures to prevent accumulation
    fig = plt.figure(figsize=figsize)
    plt.clf()  # Clear current figure
    # Don't pass selected_points to avoid blue arrow accumulation
    # User points are already included in X_sampled and will be shown in red
    plot_vectors(
        X_sampled, Y_sampled, U_sampled, V_sampled,
        title, width, height, arrow_scale=0.3, arrow_width=0.01, grid_size=grid_size, background_image=background_image, selected_points=None
    )

    # Save to file with timestamp to avoid caching issues
    import time
    timestamp = int(time.time() * 1000)
    plot_path = npy_path.replace(".npy", f"_importance_sampled_{timestamp}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=100)

    # Also convert to numpy array for direct return to Gradio
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    plot_image = np.asarray(buf)[:, :, :3]  # Extract RGB channels only

    plt.close('all')  # Close all figures after saving

    # Save sampled points to JSON (convert back to 512x512 coordinates)
    sampled_points_512 = []
    scale_x_inv = 512.0 / width
    scale_y_inv = 512.0 / height

    for i in range(len(X_sampled)):
        x_512 = int(X_sampled[i] * scale_x_inv)
        y_512 = int(Y_sampled[i] * scale_y_inv)
        # Store as handle-target pairs
        target_x_512 = int((X_sampled[i] + U_sampled[i]) * scale_x_inv)
        target_y_512 = int((Y_sampled[i] + V_sampled[i]) * scale_y_inv)

        sampled_points_512.append([x_512, y_512])
        sampled_points_512.append([target_x_512, target_y_512])

    # Save to JSON
    directory = os.path.dirname(npy_path)
    json_path = os.path.join(directory, "points_flowdrag.json")
    with open(json_path, 'w') as f:
        json.dump({"sampled_points": sampled_points_512}, f, indent=2)
    print(f"Saved {len(sampled_points_512)//2} sampled point pairs to {json_path}")

    # Return both image path and sampled points (in 512x512 coordinates)
    return plot_path, sampled_points_512


def save_intermediate_images(intermediate_images, result_dir):
    for i in range(len(intermediate_images)):
        intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
        intermediate_images_path = os.path.join(result_dir, f'output_image_{i}.png')
        cv2.imwrite(intermediate_images_path, intermediate_images[i])


def create_video(image_folder, data_folder, fps=2, first_frame_duration=2, last_frame_extra_duration=2):
    """
    Creates an MP4 video from a sequence of images using OpenCV.
    """
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    # Path to the original image with points
    data_folder = Path(data_folder)
    original_path = data_folder / 'image_with_points.jpg'
    output_path = img_folder / 'dragging.mp4'
    # Collect all image paths
    img_files = [original_path]

    # Load the first image to determine the size
    frame = cv2.imread(str(img_files[0]))
    height, width, layers = frame.shape
    size = (int(width), int(height))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
    video = cv2.VideoWriter(str(output_path), fourcc, int(fps), size)

    for _ in range(int(fps * first_frame_duration)):
        video.write(frame)

    # Add images to video
    for i in range(img_num - 2):
        video.write(cv2.imread(str(img_folder / f'output_image_{i}.png')))

    last_frame = cv2.imread(str(img_folder / 'output_image.png'))
    for _ in range(int(fps * last_frame_extra_duration)):
        video.write(last_frame)

    video.release()
    
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
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)[:,:,:3]
    
    plt.close('all')
    
    return image

def calculate_distances(handle_points, target_points):
    distances = []
    for hp, tp in zip(handle_points, target_points):
        distance = round(np.linalg.norm(np.array(hp) - np.array(tp)), 2)
        distances.append(distance)
    return distances

def save_image_all(image_folder, data_folder, points, new_points, elapsed_time, model_name):
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    # Path to the original image with points
    data_folder = Path(data_folder)
    input_image = Image.open(str(data_folder / 'input.png'))
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
    
    original_hp_tp_diff = calculate_distances(points[0], points[1])
    new_hp_tp_diff = calculate_distances(new_points[0], new_points[1])
    original_hp_tp_diff_total = sum(original_hp_tp_diff)
    new_hp_tp_diff_total = sum(new_hp_tp_diff)
    
    # concat 3 images 
    image_instruct = txt_draw(f"handle_point: {original_handle_points} \n new_handle_points: {new_handle_points} \n target_point: {target_points} \n\n hp_tp_diff: {original_hp_tp_diff} \n"
                                f"new_hp_tp_diff: {new_hp_tp_diff} \n diff_total: {original_hp_tp_diff_total:.2f} \n new_diff_total: {new_hp_tp_diff_total:.2f} \n time: {elapsed_time:.2f}s \n",
                                target_size=input_image.size)
    
    concat_image = np.concatenate([image_instruct, input_image, input_image_w_points, output_image, output_image_w_points, gt_image], axis=1)
    total = np.concatenate([text_image, concat_image], axis=0)
            
    Image.fromarray(total).save(str(img_folder / 'concat_image.png'))

    return {
        "handle_points": original_handle_points,
        "new_handle_points": new_handle_points,
        "target_points": target_points,
        "hp_tp_diff": original_hp_tp_diff,
        "new_hp_tp_diff": new_hp_tp_diff,
        "diff_total": original_hp_tp_diff_total,
        "new_diff_total": new_hp_tp_diff_total,
    }

def run_gooddrag(source_image,
                 mask,
                 prompt,
                 points,
                 inversion_strength,
                 lam,
                 latent_lr,
                 model_path,
                 vae_path,
                 lora_path,
                 drag_end_step,
                 track_per_step,
                 r1,
                 r2,
                 d,
                 max_drag_per_track,
                 max_track_no_change,
                 feature_idx=3,
                 result_save_path='',
                 return_intermediate_images=True,
                 vector_field_path=None, # npy_path
                 drag_loss_threshold=0,
                 save_intermedia=False,
                 compare_mode=False,
                 once_drag=False,
                 ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    n_inference_step = 50
    guidance_scale = 1.0
    seed = 42
    dragger = FlowDrag(device, model_path, prompt, height, width, inversion_strength, r1, r2, d,
                          drag_end_step, track_per_step, lam, latent_lr,
                          n_inference_step, guidance_scale, feature_idx, compare_mode, vae_path, lora_path, seed,
                          max_drag_per_track, drag_loss_threshold, once_drag, max_track_no_change)

    source_image = preprocess_image(source_image, device)
    
    #NOTE: good_drag
    gen_image, intermediate_features, new_points_handle, intermediate_images = \
        dragger.good_drag(source_image, points,
                          mask,
                          return_intermediate_images=return_intermediate_images)

    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename='drag_image')

    gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f'points {points}')
    print(f'new points {new_points}')

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode='bilinear')
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
            intermediate_images_path = os.path.join(result_save_path, f'output_image_{i}.png')
            cv2.imwrite(intermediate_images_path, intermediate_images[i])

    return out_image, new_points


def run_flowdrag(source_image,
                 mask,
                 prompt,
                 points,
                 inversion_strength,
                 lam,
                 latent_lr,
                 model_path,
                 vae_path,
                 lora_path,
                 drag_end_step,
                 track_per_step,
                 r1,
                 r2,
                 d,
                 max_drag_per_track,
                 max_track_no_change,
                 feature_idx=3,
                 result_save_path='',
                 return_intermediate_images=True,
                 vector_field_path=None, # npy_path
                 drag_loss_threshold=0,
                 save_intermedia=False,
                 compare_mode=False,
                 once_drag=False,
                 ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    n_inference_step = 50
    guidance_scale = 1.0
    seed = 42
    dragger = FlowDrag(device, model_path, prompt, height, width, inversion_strength, r1, r2, d,
                          drag_end_step, track_per_step, lam, latent_lr,
                          n_inference_step, guidance_scale, feature_idx, compare_mode, vae_path, lora_path, seed,
                          max_drag_per_track, drag_loss_threshold, once_drag, max_track_no_change)

    source_image = preprocess_image(source_image, device)
    
    # Load the vector field from the .npy file
    # For evaluation, vector_field_path may be None if flowdrag points are provided by json file
    if vector_field_path is not None:
        vector_field = torch.tensor(np.load(vector_field_path)).to(device)
        # height, width, _ = vector_field.shape
    else:
        vector_field = None

    #NOTE: flow_drag with optimized hierarchical supervision
    # Calculate num_user_points: assume first point pair is from user, rest are from vector field sampling
    # Each point pair = (handle_x, handle_y, target_x, target_y), so 4 values per pair
    # Actually points format is [x1, y1, x2, y2, ...] where odd indices are handle, even are target
    # So num_pairs = len(points) // 4, and typically first 1 pair is user-provided
    num_user_points = 1  # Default: first point pair is from user

    # If there are very few points (<=2 pairs = 8 values), assume all are user points
    if len(points) <= 8:
        num_user_points = len(points) // 4

    print(f"run_flowdrag: Total point pairs={len(points)//4}, User point pairs={num_user_points}, VF point pairs={len(points)//4 - num_user_points}")

    gen_image, intermediate_features, new_points_handle, intermediate_images = \
        dragger.flow_drag(source_image, points,
                          mask,
                          return_intermediate_images=return_intermediate_images,
                          num_user_points=num_user_points)

    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename='drag_image')

    gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f'points {points}')
    print(f'new points {new_points}')

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode='bilinear')
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
            intermediate_images_path = os.path.join(result_save_path, f'output_image_{i}.png')
            cv2.imwrite(intermediate_images_path, intermediate_images[i])

    return out_image, new_points
