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
from utils.ui_utils import (
    get_points, undo_points, show_cur_points,
    clear_all, store_img, train_lora_interface, run_gooddrag, save_image_mask_points, save_image_mask_points_concat, save_drag_result, save_image_mask_points_concat_flowdrag,
    save_intermediate_images, create_video
)

css = """
canvas, .svelte-1ipelgc canvas {
    object-fit: contain !important;  /* 이미지가 잘리지 않고 온전히 보이도록 설정 */
    background: transparent !important;
    width: 80% !important;
    height: 80% !important;
}
"""

LENGTH = 450
SD_15_PATH = "/mnt/hdd/sunjaeyoon/workspace/pretrain_SD_models/CompVis/stable-diffusion-v1-5"
SD_21_PATH = "/mnt/hdd/sunjaeyoon/workspace/pretrain_SD_models/stabilityai/stable-diffusion-2-1-base"

def create_markdown_section():
    gr.Markdown("""
# FlowDrag ✨

👋 Dataset Labeling 🌟

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
                                     choices=[
                                                SD_21_PATH,
                                                 "runwayml/stable-diffusion-v1-5",
                                                 "stabilityai/stable-diffusion-2-1-base",
                                                 "stabilityai/stable-diffusion-xl-base-1.0",
                                             ] + local_models_choice
                                     )
            vae_path = gr.Dropdown(value="stabilityai/sd-vae-ft-mse",
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


def create_real_image_editing_ui():
    scale = 0.8
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>📤 Original Image</h2>")
            original_image_unedit = gr.Image(type="numpy", label="Original Image",
                              show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
            # canvas = gr.Image(type="numpy", tool="sketch", label="Draw your mask on the image",
            #                   show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
            with gr.Row():
                train_lora_button = gr.Button("Train LoRA")
                lora_path = gr.Textbox(value=f"./lora_data/test", label="LoRA Path",
                                       placeholder="Enter path for LoRA data")

            with gr.Row():
                lora_status_bar = gr.Textbox(label="LoRA Training Status", interactive=False)

        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>✏️ Draw Mask in Blended Image</h2>")
            canvas = gr.Image(type="numpy", tool="sketch", label="Draw your mask on the image",
                              show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
            # input_image = gr.Image(type="numpy", label="Click on the image to mark points",
            #                        show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
            # with gr.Row():
            #     undo_button = gr.Button("Undo Point")
            #     save_button = gr.Button('Save Current Data')
            #     data_dir = gr.Textbox(value='./VFD_Bench_Dataset/', label="Data Directory",
            #                           placeholder="Enter directory path for mask and points")
                
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🍰 Click Points</h2>")
            input_image = gr.Image(type="numpy", label="Click on the image to mark points",
                                   show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
            with gr.Row():
                undo_button = gr.Button("Undo Point")
                # save_button = gr.Button('Save Current Data')
            with gr.Row():
                data_dir = gr.Textbox(value='./VFD_Bench_Dataset/', label="Data Directory",
                                      placeholder="Enter directory path for mask and points")

        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🌟 GT</h2>")
            gt_image = gr.Image(type="numpy", label="GT Image",
                                   show_label=True, height=LENGTH, width=LENGTH)  # for points clicking
            with gr.Row():
                save_concat_button = gr.Button('Save Concat Image')
                # data_dir = gr.Textbox(value='./VFD_Bench_Dataset/', label="Data Directory",
                #                       placeholder="Enter directory path for mask and points")
            with gr.Row():
                save_concat_flowdrag_button = gr.Button('Save Concat Image FlowDrag')
                # data_dir = gr.Textbox(value='./dataset/test', label="Data Directory",
                #                       placeholder="Enter directory path for mask and points")
    
    # with gr.Row():
    #     with gr.Column():
    #         gr.Markdown("<h2 style='text-align: center;'>🖼️ Editing Result</h2>")
    #         output_image = gr.Image(type="numpy", label="View the editing results here",
    #                                 show_label=True, height=LENGTH, width=LENGTH)
    #         with gr.Row():
    #             run_button = gr.Button("Run")
    #             clear_all_button = gr.Button("Clear All")
    #             save_result = gr.Button("Save Result")
    #             show_points = gr.Button("Show Points")
    #             result_save_path = gr.Textbox(value='./result/test', label="Result Folder",
    #                                           placeholder="Enter path to save the results")

    # return canvas, train_lora_button, lora_path, lora_status_bar, input_image, gt_image, undo_button, save_button, save_concat_button, data_dir, \
    #        output_image, run_button, clear_all_button, show_points, result_save_path, save_result

    return original_image_unedit, canvas, train_lora_button, lora_path, lora_status_bar, input_image, gt_image, undo_button, \
        save_concat_button, data_dir, save_concat_flowdrag_button

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


def attach_canvas_event(canvas: gr.State, original_image: gr.State,
                        selected_points: gr.State, input_image, mask):
    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    
def attach_blend_canvas_event(blend_canvas: gr.State, blend_original_image: gr.State,
                              selected_points: gr.State, blend_image, mask):
    blend_canvas.edit(
        store_img,
        [blend_canvas],
        [blend_original_image, selected_points, blend_image, mask]
    )   

def attach_input_image_event(input_image, selected_points):
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image]
    )
    
#NOTE: added by kookie 25.05.30 for dataset labeling -> alpha blending
def attach_blend_image_event(blend_image, selected_points):
    blend_image.select(
        get_points,
        [blend_image, selected_points],
        [blend_image]
    )

def attach_undo_button_event(undo_button, original_image, selected_points, mask, input_image):
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points],
    )

#NOTE: added by kookie 25.05.30 for dataset labeling -> alpha blending
def attach_undo_blend_button_event(undo_blend_button, original_image, selected_points, mask, blend_image):
    undo_blend_button.click(
        undo_points,
        [original_image, mask],
        [blend_image, selected_points]
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


def attach_run_button_event(run_button, original_image, input_image, mask, prompt,
                            selected_points, inversion_strength, lam, latent_lr,
                            model_path, vae_path, lora_path,
                            drag_end_step, drag_per_step,
                            output_image, r1, r2, d, feature_idx, new_points,
                            max_drag_per_track, max_track_no_change,
                            result_save_path, save_intermediates_images):
    run_button.click(
        run_gooddrag,
        [original_image, input_image, mask, prompt, selected_points,
         inversion_strength, lam, latent_lr, model_path, vae_path,
         lora_path, drag_end_step, drag_per_step, r1, r2, d,
         max_drag_per_track, max_track_no_change, feature_idx, result_save_path, save_intermediates_images],
        [output_image, new_points]
    )


def attach_show_points_event(show_points, output_image, selected_points):
    show_points.click(
        show_cur_points,
        [output_image, selected_points],
        [output_image]
    )


def attach_clear_all_button_event(clear_all_button, canvas, input_image,
                                  output_image, selected_points, original_image, mask):
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas, input_image, output_image, selected_points, original_image, mask]
    )


def attach_save_button_event(save_button, mask, selected_points, input_image, save_dir):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_button.click(
        save_image_mask_points,
        inputs=[mask, selected_points, input_image, save_dir],
        outputs=[]
    )

#NOTE: added by kookie 25.05.28 for dataset labeling -> concat된 이미지까지 저장해줌
def attach_save_concat_button_event(save_concat_button, original_image_unedit, mask, selected_points, input_image, gt_image, save_dir):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_concat_button.click(
        save_image_mask_points_concat,
        inputs=[original_image_unedit, mask, selected_points, input_image, gt_image, save_dir],
        outputs=[]
    )

#NOTE: added by kookie 25.05.29 for dataset labeling -> concat된 이미지까지 저장해줌 FlowDrag 전용!!
def attach_save_concat_flowdrag_button_event(save_concat_flowdrag_button, original_image_unedit, mask, selected_points, input_image, gt_image, save_dir):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_concat_flowdrag_button.click(
        save_image_mask_points_concat_flowdrag,
        inputs=[original_image_unedit, mask, selected_points, input_image, gt_image, save_dir],
        outputs=[]
    )
    
def attach_save_result_event(save_result, output_image, new_points, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_result.click(
        save_drag_result,
        inputs=[output_image, new_points, result_path],
        outputs=[]
    )
    
#NOTE: added by kookie 
def attach_save_result_kookie_event(save_result, input_image, output_image, new_points, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_result.click(
        save_drag_result,
        inputs=[output_image, new_points, result_path],
        outputs=[]
    )


def attach_video_event(get_mp4_button, result_save_path, data_dir):
    get_mp4_button.click(
        create_video,
        inputs=[result_save_path, data_dir]
    )


def main():
    with gr.Blocks(css=css) as demo:
        mask = gr.State(value=None)
        selected_points = gr.State([])
        new_points = gr.State([])
        original_image = gr.State(value=None)
        create_markdown_section()
        intermediate_images = gr.State([])
        blend_original_image = gr.State(value=None) #NOTE: added by kookie 25.05.30 for dataset labeling -> alpha blending

        # canvas, train_lora_button, lora_path, lora_status_bar, input_image, gt_image, undo_button, save_button, save_concat_button, data_dir, \
        # output_image, run_button, clear_all_button, show_points, result_save_path, \
        # save_result = create_real_image_editing_ui()

        original_image_unedit, canvas, train_lora_button, lora_path, lora_status_bar, input_image, gt_image, undo_button, save_concat_button, data_dir, save_concat_flowdrag_button = create_real_image_editing_ui()

        latent_lr, prompt, drag_end_step, drag_per_step = create_drag_parameters_ui()

        model_path, vae_path = create_base_model_config_ui()
        lora_step, lora_lr, lora_batch_size, lora_rank = create_lora_parameters_ui()
        r1, r2, d, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change = \
            create_advance_parameters_ui()
        save_intermediates_images, get_mp4_button = create_intermediate_save_ui()

        attach_canvas_event(canvas, original_image, selected_points, input_image, mask)
        # attach_blend_canvas_event(blend_canvas, blend_original_image, selected_points, blend_image, mask) #NOTE: added by kookie 25.05.30 for dataset labeling -> alpha blending
        attach_input_image_event(input_image, selected_points)
        # attach_blend_image_event(blend_image, selected_points)
        attach_undo_button_event(undo_button, original_image, selected_points, mask, input_image)
        # attach_undo_blend_button_event(undo_button, blend_original_image, selected_points, mask, blend_image) #NOTE: added by kookie 25.05.30 for dataset labeling -> alpha blending
        
        attach_train_lora_button_event(train_lora_button, original_image, prompt, model_path, vae_path, lora_path,
                                       lora_step, lora_lr, lora_batch_size, lora_rank, lora_status_bar)
        # attach_run_button_event(run_button, original_image, input_image, mask, prompt, selected_points,
        #                         inversion_strength, lam, latent_lr, model_path, vae_path, lora_path,
        #                         drag_end_step, drag_per_step, output_image,
        #                         r1, r2, d, feature_idx, new_points, max_drag_per_track,
        #                         max_track_no_change, result_save_path, save_intermediates_images)
        # attach_show_points_event(show_points, output_image, new_points)
        # attach_clear_all_button_event(clear_all_button, canvas, input_image, output_image, selected_points,
        #                               original_image, mask)
        # attach_save_button_event(save_button, mask, selected_points, input_image, data_dir)
        attach_save_concat_button_event(save_concat_button, original_image_unedit, mask, selected_points, input_image, gt_image, data_dir)
        attach_save_concat_flowdrag_button_event(save_concat_flowdrag_button, original_image_unedit, mask, selected_points, input_image, gt_image, data_dir)
        # attach_save_result_event(save_result, output_image, new_points, result_save_path)
        # attach_save_result_kookie_event(save_result, input_image, output_image, new_points, result_save_path)
        # attach_video_event(get_mp4_button, result_save_path, data_dir)

    demo.queue().launch(share=True, debug=True)


if __name__ == '__main__':
    main()
