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
import gradio as gr
from PIL import Image
import numpy as np
from utils.ui_utils import (
    get_points, undo_points, show_cur_points,
    clear_all, store_img, train_lora_interface, run_gooddrag, save_image_mask_points, save_drag_result,
    save_intermediate_images, create_video, kookie_save_result, load_and_visualize_vector_field, normal_sampling_vector_field, importance_sampling_vector_field, run_flowdrag
)

LENGTH = 512
# Default model paths - users can modify these or set via environment variables
SD_15_PATH = os.getenv("SD_15_PATH", "runwayml/stable-diffusion-v1-5")
SD_21_PATH = os.getenv("SD_21_PATH", "stabilityai/stable-diffusion-2-1-base")

def create_markdown_section():
    gr.Markdown("""
# FlowDrag 
🌊 Welcome to FlowDrag! Leverage optical flow for precise and automated image manipulation.

## Quick Start Guide

### 📤 Step 1: Upload Your Image
Simply drag and drop an image into the **Draw Mask** box, or click to browse your files.

### 🛠️ Step 2: Train LoRA (One-time Setup)
- Specify the LoRA save path
- Click **Train LoRA** to adapt the diffusion model to your image
- *This ensures high-quality results tailored to your specific image*

### ✏️ Step 3: Define the Manipulation Region
Draw a mask to specify which parts of the image should be manipulated.

### 🌊 Step 4: Upload Vector Field
Upload a `.npy` file containing optical flow data:
- The vector field encodes motion patterns across your image
- Each vector represents the desired displacement at that pixel location
- You can generate this using optical flow tools or define custom motion patterns

### 🎯 Step 5: Automatic Point Sampling
Instead of manually clicking points, FlowDrag automatically extracts them:
- **Sampling Number**: How many handle-target point pairs to extract
- **Grid Size**: Controls spatial distribution of samples
- **Normal Sampling**: Uniformly distributed points across the masked region
- **Importance Sampling**: Prioritizes areas with larger motion (higher vector magnitudes)

*Click the sampling button to visualize extracted points on your image!*

### ▶️ Step 6: Run FlowDrag
Hit **Run** to start the manipulation process. FlowDrag will:
- Use sampled points as supervision
- Apply motion guided by the vector field

    """)

def create_base_model_config_ui():
    with gr.Tab("Diffusion Model"):
        with gr.Row():
            local_models_dir = 'local_pretrained_models'
            os.makedirs(local_models_dir, exist_ok=True)
            local_models_choice = \
                [os.path.join(local_models_dir, d) for d in os.listdir(local_models_dir) if
                 os.path.isdir(os.path.join(local_models_dir, d))]
            model_path = gr.Dropdown(value=SD_21_PATH, # "runwayml/stable-diffusion-v1-5",
                                     label="Diffusion Model Path",
                                     choices=[  SD_21_PATH,
                                                 "runwayml/stable-diffusion-v1-5",
                                                 "stabilityai/stable-diffusion-2-1-base",
                                                 "stabilityai/stable-diffusion-xl-base-1.0",
                                             ] + local_models_choice
                                     )
            vae_path = gr.Dropdown(value="default", # "stabilityai/sd-vae-ft-mse",
                                   label="VAE choice",
                                   choices=["stabilityai/sd-vae-ft-mse",
                                            "default"] + local_models_choice
                                   )

    return model_path, vae_path


def create_lora_parameters_ui():
    with gr.Tab("LoRA Parameters"):
        with gr.Row():
            lora_step = gr.Number(value=70, label="LoRA training steps", precision=0)
            lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
            lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
            lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)

    return lora_step, lora_lr, lora_batch_size, lora_rank

def adjust_canvas_size(image):
    height, width, _ = image.shape  # 이미지의 실제 크기 가져오기
    return width, height

def load_image_from_path(image_path):
    """Load an image from the given path and return as numpy array"""
    try:
        if not os.path.exists(image_path):
            return None
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((LENGTH, LENGTH))
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def create_real_image_editing_ui():
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>📤 Draw Mask</h2>")            
            canvas = gr.Image(type="numpy", tool="sketch", label="Draw your mask on the image",
                              show_label=True, height=LENGTH, width=LENGTH)  # shape=(LENGTH, LENGTH) for mask painting
            
            with gr.Row():
                image_path_input = gr.Textbox(value='./samples/dog_flower/dog_flower.jpg', 
                                              label="Image Path", 
                                              placeholder="Enter path to image file",
                                              scale=3)
                upload_image_button = gr.Button("Upload", scale=1)
            
            with gr.Row():
                train_lora_button = gr.Button("Train LoRA")
                lora_path = gr.Textbox(value=f"./lora_data/dog_flower", label="LoRA Path", #NOTE: LORA Path
                                       placeholder="Enter path for LoRA data")

            with gr.Row():
                lora_status_bar = gr.Textbox(label="LoRA Training Status", interactive=False)

        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🍭 Click Points</h2>") 
            input_image = gr.Image(type="numpy", label="Click on the image to mark points",
                                   show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
            with gr.Row():
                undo_button = gr.Button("Undo Point")
                load_selected_button = gr.Button('Load Points')
                save_button = gr.Button('Save Current Data')
                data_dir = gr.Textbox(value='./dataset/test', label="Data Directory",
                                      placeholder="Enter directory path for mask and points")
                selected_points_message = gr.Textbox(label="Selected Points", interactive=False)
                # save_intermediate_button = gr.Button("Save Intermediate Video")
                
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>📂 Load and Visualize Vector Field</h2>")
            vector_field_image = gr.Image(label="Vector Field Visualization", interactive=False, show_label=True, height=LENGTH, width=LENGTH, fit="contain")
            npy_path = gr.Textbox(value='./samples/dog_flower/2d_vector_flow_field.npy', label="npy path", placeholder="Enter the path to the npy file")
            visualize_button = gr.Button("Vector Field Visualization")
            result_message = gr.Textbox(label="Result", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🥕 Sampling 2D Vector Field</h2>")
            sampling_vector_field_image = gr.Image(label="Sampling Visualization", interactive=False, show_label=True, height=LENGTH, width=LENGTH, fit="contain")
            sampling_num = gr.Textbox(value='25', label="sampling number", placeholder="Enter the sampling number")
            grid_size = gr.Textbox(value='30', label="grid size", placeholder="Enter the grid size N")
            normal_sampling_button = gr.Button("Normal Sampling (Save Points)")
            importance_sampling_button = gr.Button("Importance Sampling (Save Points)")
            final_selected_points_message = gr.Textbox(label="Final Sampled Points", interactive=False)
        
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🌇 Editing Result (GoodDrag)</h2>")
            output_image_gooddrag = gr.Image(type="numpy", label="View the editing results here",
                                    show_label=True, height=LENGTH, width=LENGTH)
            with gr.Row():
                run_button = gr.Button("Run")
                show_points = gr.Button("Show Points")
                
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🖼️ Editing Result (FlowDrag)</h2>")
            output_image_flowdrag = gr.Image(type="numpy", label="View the editing results here",
                                    show_label=True, height=LENGTH, width=LENGTH)
            with gr.Row():
                run_flowdrag_button = gr.Button("Run")
                clear_all_button = gr.Button("Clear All")
                show_points_flowdrag = gr.Button("Show Points")
                flowdrag_save_all_button = gr.Button("Save All !!")
                result_save_path = gr.Textbox(value='./result/dog_flower/', label="Result Folder",
                                              placeholder="Enter path to save the results")

    return canvas, image_path_input, upload_image_button, train_lora_button, lora_path, lora_status_bar, input_image, undo_button, load_selected_button, save_button, data_dir, selected_points_message, \
           output_image_gooddrag, output_image_flowdrag, run_button, clear_all_button, show_points, result_save_path, \
           npy_path, visualize_button, normal_sampling_button, importance_sampling_button, sampling_num, grid_size, \
           result_message, vector_field_image, sampling_vector_field_image, final_selected_points_message, \
           show_points_flowdrag, run_flowdrag_button, flowdrag_save_all_button 


def create_drag_parameters_ui():
    with gr.Tab("Drag Parameters"):
        with gr.Row():
            latent_lr = gr.Number(value=0.02, label="Learning rate")
            prompt = gr.Textbox(label="Prompt")
            drag_end_step = gr.Number(value=7, label="End time step", precision=0)
            drag_per_step = gr.Number(value=10, label="Point tracking number per each step", precision=0)

    return latent_lr, prompt, drag_end_step, drag_per_step


def create_advance_parameters_ui():
    with gr.Tab("Advanced Parameters"):
        with gr.Row():
            r1 = gr.Number(value=4, label="Motion supervision feature path size", precision=0)
            r2 = gr.Number(value=12, label="Point tracking feature patch size", precision=0)
            drag_distance = gr.Number(value=4, label="The distance for motion supervision", precision=0)
            feature_idx = gr.Number(value=3, label="The index of the features [0,3]", precision=0)
            max_drag_per_track = gr.Number(value=3,
                                           label="Motion supervision times for each point tracking",
                                           precision=0)

        with gr.Row():
            lam = gr.Number(value=0.2, label="Lambda", info="Regularization strength on unmasked areas")
            inversion_strength = gr.Slider(0, 1.0,
                                           value=0.75,
                                           label="Inversion strength")
            max_track_no_change = gr.Number(value=10, label="Early stop",
                                            info="The maximum number of times points is unchanged.")

    return (r1, r2, drag_distance, feature_idx, max_drag_per_track, lam,
            inversion_strength, max_track_no_change)


def create_intermediate_save_ui():
    with gr.Tab("Get Intermediate Images"):
        with gr.Row():
            save_intermediates_images = gr.Checkbox(label='Save intermediate images')
            get_mp4 = gr.Button("Get video")

    return save_intermediates_images, get_mp4


def attach_upload_image_button_event(upload_image_button, image_path_input, canvas):
    """Attach event to upload button to load image from path"""
    upload_image_button.click(
        load_image_from_path,
        inputs=[image_path_input],
        outputs=[canvas]
    )

def attach_canvas_event(canvas: gr.State, original_image: gr.State,
                        user_selected_points: gr.State, input_image, mask):
    canvas.edit(
        store_img,
        [canvas],
        [original_image, user_selected_points, input_image, mask]
    )

def attach_input_image_event(input_image, user_selected_points):
    input_image.select(
        get_points,
        [input_image, user_selected_points],
        [input_image]
    )


def attach_undo_button_event(undo_button, original_image, user_selected_points, mask, input_image):
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, user_selected_points]
    )


def attach_train_lora_button_event(train_lora_button, original_image, prompt,
                                   model_path, vae_path, lora_path,
                                   lora_step, lora_lr, lora_batch_size, lora_rank,
                                   lora_status_bar):
    train_lora_button.click(
        train_lora_interface,
        [original_image, prompt, model_path, vae_path, lora_path,
         lora_step, lora_lr, lora_batch_size, lora_rank],
        [lora_status_bar]
    )


def attach_run_gooddrag_button_event(run_button, original_image, input_image, mask, prompt,
                            final_selected_points, inversion_strength, lam, latent_lr,
                            model_path, vae_path, lora_path,
                            drag_end_step, drag_per_step,
                            output_image_gooddrag, r1, r2, d, feature_idx, new_points_gooddrag,
                            max_drag_per_track, max_track_no_change,
                            result_save_path, save_intermediates_images,
                            npy_path):
    run_button.click(
        run_gooddrag, #NOTE: run_gooddrag
        [original_image, mask, prompt, final_selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, result_save_path, save_intermediates_images, npy_path],
        [output_image_gooddrag, new_points_gooddrag]
    )

def attach_run_flowdrag_button_event(run_flowdrag_button, original_image, mask, prompt,
                            final_selected_points, inversion_strength, lam, latent_lr,
                            model_path, vae_path, lora_path,
                            drag_end_step, drag_per_step,
                            output_image_flowdrag, r1, r2, d, feature_idx, new_points_flowdrag,
                            max_drag_per_track, max_track_no_change,
                            result_save_path, save_intermediates_images,
                            npy_path):
    run_flowdrag_button.click(
        run_flowdrag, #NOTE: run_flowdrag
        [original_image, mask, prompt, final_selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, result_save_path, save_intermediates_images, npy_path],
        [output_image_flowdrag, new_points_flowdrag]
    )

def attach_show_points_event_gooddrag(show_points_button, output_image_gooddrag, new_points_gooddrag):
    show_points_button.click(
        show_cur_points,
        [output_image_gooddrag, new_points_gooddrag],
        [output_image_gooddrag]
    )

def attach_show_points_event_flowdrag(show_points_button_flowdrag, output_image_flowdrag, new_points_flowdrag):
    show_points_button_flowdrag.click(
        show_cur_points,
        [output_image_flowdrag, new_points_flowdrag],
        [output_image_flowdrag]
    )
    
def attach_clear_all_button_event(clear_all_button, canvas, input_image,
                                  output_image_gooddrag, output_image_flowdrag, vector_field_image, user_selected_points, original_image, mask):
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas, input_image, output_image_gooddrag, output_image_flowdrag, vector_field_image, user_selected_points, original_image, mask]
    )


def attach_save_button_event(save_button, mask, user_selected_points, input_image, save_dir):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_button.click(
        save_image_mask_points,
        inputs=[mask, user_selected_points, input_image, save_dir],
        outputs=[]
    )

#NOTE: save all results
def attach_save_gooddrag_result_event(gooddrag_save_all_button, mask, user_selected_points, input_image, output_gooddrag_image, new_points, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    gooddrag_save_all_button.click(
        kookie_save_result,
        inputs=[mask, user_selected_points, input_image, output_gooddrag_image, new_points, result_path],
        outputs=[]
    )

def attach_save_flowdrag_result_event(flowdrag_save_all_button, mask, user_selected_points, input_image, output_image_gooddrag, new_points_gooddrag, output_flowdrag_image, new_points_flowdrag, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    flowdrag_save_all_button.click(
        kookie_save_result,
        inputs=[mask, user_selected_points, input_image, output_image_gooddrag, new_points_gooddrag, output_flowdrag_image, new_points_flowdrag, result_path],
        outputs=[]
    )


def attach_video_event(get_mp4_button, result_save_path, data_dir): # get_mp4_button, result_save_path, data_dir
    get_mp4_button.click(
        create_video,
        inputs=[result_save_path, result_save_path]
    )

def attach_vector_field_visualization_event(visualize_button, npy_path, result_message, vector_field_image, original_image, user_selected_points):
    visualize_button.click(
        load_and_visualize_vector_field,
        inputs=[npy_path, original_image, user_selected_points],
        outputs=[result_message, vector_field_image]
    )

def attach_normal_sampling_vector_field_event(normal_sampling_button, npy_path, sampling_num, grid_size, sampling_vector_field_image, original_image, user_selected_points, final_selected_points, final_selected_points_message):
    """
    Attach normal sampling event that returns visualization image, final points, and message.
    The sampled points are NOT accumulated - each sampling creates fresh results.
    """
    def normal_sampling_with_point_update(npy_path, sampling_num, grid_size, uploaded_image, current_user_points):
        plot_image, sampled_points_512 = normal_sampling_vector_field(
            npy_path, sampling_num, grid_size, uploaded_image, current_user_points
        )
        # Create readable message for display
        num_pairs = len(sampled_points_512) // 2
        message = f"Sampled {num_pairs} point pairs: {sampled_points_512}"
        print(f"Normal sampling: Sampled {num_pairs} point pairs")
        return plot_image, sampled_points_512, message

    normal_sampling_button.click(
        normal_sampling_with_point_update,
        inputs=[npy_path, sampling_num, grid_size, original_image, user_selected_points],
        outputs=[sampling_vector_field_image, final_selected_points, final_selected_points_message]
    )

def attach_importance_sampling_vector_field_event(importance_sampling_button, npy_path, sampling_num, grid_size, sampling_vector_field_image, original_image, user_selected_points, final_selected_points, final_selected_points_message):
    """
    Attach importance sampling event that returns visualization image, final points, and message.
    The sampled points are NOT accumulated - each sampling creates fresh results.
    """
    def importance_sampling_with_point_update(npy_path, sampling_num, grid_size, uploaded_image, current_user_points):
        plot_image, sampled_points_512 = importance_sampling_vector_field(
            npy_path, sampling_num, grid_size, clip_threshold=5.0,
            uploaded_image=uploaded_image, selected_points=current_user_points
        )
        # Create readable message for display
        num_pairs = len(sampled_points_512) // 2
        message = f"Sampled {num_pairs} point pairs: {sampled_points_512}"
        print(f"Importance sampling: Sampled {num_pairs} point pairs")
        return plot_image, sampled_points_512, message

    importance_sampling_button.click(
        importance_sampling_with_point_update,
        inputs=[npy_path, sampling_num, grid_size, original_image, user_selected_points],
        outputs=[sampling_vector_field_image, final_selected_points, final_selected_points_message]
    )


def attach_selected_points_event(load_selected_button, user_selected_points, selected_points_message):

    def get_points(user_points):
        return str(user_points)

    load_selected_button.click(
        get_points,
        inputs=[user_selected_points],
        outputs=[selected_points_message]
    )

def main():
    with gr.Blocks() as demo:
        mask = gr.State(value=None)
        user_selected_points = gr.State([])
        final_selected_points = gr.State([])  # Points after sampling (for model input)
        new_points_gooddrag = gr.State([])
        new_points_flowdrag = gr.State([])
        original_image = gr.State(value=None)
        create_markdown_section()
        intermediate_images = gr.State([])

        canvas, image_path_input, upload_image_button, train_lora_button, lora_path, lora_status_bar, input_image, undo_button, load_selected_button, save_button, data_dir, selected_points_message,\
        output_image_gooddrag, output_image_flowdrag, run_button, clear_all_button, show_points, result_save_path, \
        npy_path, visualize_button, normal_sampling_button, importance_sampling_button, sampling_num, grid_size, \
        result_message, vector_field_image, sampling_vector_field_image, final_selected_points_message, \
        show_points_flowdrag, run_flowdrag_button, flowdrag_save_all_button = create_real_image_editing_ui() 

        latent_lr, prompt, drag_end_step, drag_per_step = create_drag_parameters_ui()

        model_path, vae_path = create_base_model_config_ui()
        lora_step, lora_lr, lora_batch_size, lora_rank = create_lora_parameters_ui()
        r1, r2, d, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change = \
            create_advance_parameters_ui()
        save_intermediates_images, get_mp4_button = create_intermediate_save_ui()

        attach_upload_image_button_event(upload_image_button, image_path_input, canvas)
        attach_canvas_event(canvas, original_image, user_selected_points, input_image, mask)
        attach_input_image_event(input_image, user_selected_points)
        attach_undo_button_event(undo_button, original_image, user_selected_points, mask, input_image)
        attach_train_lora_button_event(train_lora_button, original_image, prompt, model_path, vae_path, lora_path,
                                       lora_step, lora_lr, lora_batch_size, lora_rank, lora_status_bar)

        attach_selected_points_event(load_selected_button, user_selected_points, selected_points_message)

        #NOTE: gooddrag - uses user_selected_points
        attach_run_gooddrag_button_event(run_button, original_image, input_image, mask, prompt, user_selected_points,
                                inversion_strength, lam, latent_lr, model_path, vae_path, lora_path,
                                drag_end_step, drag_per_step, output_image_gooddrag,
                                r1, r2, d, feature_idx, new_points_gooddrag, max_drag_per_track,
                                max_track_no_change, result_save_path, save_intermediates_images,
                                npy_path)

        #NOTE: flowdrag - uses final_selected_points from sampling
        attach_run_flowdrag_button_event(run_flowdrag_button, original_image, mask, prompt, final_selected_points,
                                inversion_strength, lam, latent_lr, model_path, vae_path, lora_path,
                                drag_end_step, drag_per_step, output_image_flowdrag,
                                r1, r2, d, feature_idx, new_points_flowdrag, max_drag_per_track,
                                max_track_no_change, result_save_path, save_intermediates_images,
                                npy_path)

        attach_show_points_event_gooddrag(show_points, output_image_gooddrag, new_points_gooddrag)
        attach_show_points_event_flowdrag(show_points_flowdrag, output_image_flowdrag, new_points_flowdrag)
        attach_clear_all_button_event(clear_all_button, canvas, input_image, output_image_gooddrag, output_image_flowdrag, vector_field_image, user_selected_points,
                                      original_image, mask)
        attach_save_button_event(save_button, mask, user_selected_points, input_image, data_dir)
        # attach_save_result_event(save_result, output_image, new_points, result_save_path)
        attach_video_event(get_mp4_button, result_save_path, data_dir)
        attach_vector_field_visualization_event(visualize_button, npy_path, result_message, vector_field_image, original_image, user_selected_points)
        # attach_save_gooddrag_result_event(gooddrag_save_all_button, mask, user_selected_points, input_image, output_image, new_points_gooddrag, result_save_path)

        attach_normal_sampling_vector_field_event(normal_sampling_button, npy_path, sampling_num, grid_size, sampling_vector_field_image, original_image, user_selected_points, final_selected_points, final_selected_points_message)
        attach_importance_sampling_vector_field_event(importance_sampling_button, npy_path, sampling_num, grid_size, sampling_vector_field_image, original_image, user_selected_points, final_selected_points, final_selected_points_message)

        attach_save_flowdrag_result_event(flowdrag_save_all_button, mask, user_selected_points, input_image, output_image_gooddrag, new_points_gooddrag, output_image_flowdrag, new_points_flowdrag, result_save_path)

    demo.queue().launch(share=True, debug=True)


if __name__ == '__main__':
    main()
