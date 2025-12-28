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


import torch
import numpy as np
import copy

import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils.drag_utils import point_tracking, check_handle_reach_target, interpolate_feature_patch, point_tracking_kookie
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from diffusers import DDIMScheduler, AutoencoderKL
from pytorch_lightning import seed_everything
from accelerate import Accelerator
from tqdm import tqdm
import itertools
import pdb
# from  diffusers.models.attention_processor import LoRAAttnProcessor2_0


# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):
    def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
            last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


class FlowDrag:
    def __init__(self, device, model_path: str, prompt: str,
                 full_height: int, full_width: int,
                 inversion_strength: float,
                 r1: int = 4, r2: int = 12, beta: int = 4,
                 drag_end_step: int = 10, track_per_denoise: int = 10,
                 lam: float = 0.2, latent_lr: float = 0.01,
                 n_inference_step: int = 50, guidance_scale: float = 1.0, feature_idx: int = 3,
                 compare_mode: bool = False,
                 vae_path: str = "default", lora_path: str = '', seed: int = 42,
                 max_drag_per_track: int = 10, drag_loss_threshold: float = 4.0, once_drag: bool = False,
                 max_track_no_change: int = 10):
        self.device = device
        self.vae_path = vae_path
        self.lora_path = lora_path
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                  beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, steps_offset=1)

        is_sdxl = 'xl' in model_path
        self.is_sdxl = is_sdxl
        if is_sdxl:
            self.model = StableDiffusionXLPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
            self.model.unet.config.addition_embed_type = None
        else:
            self.model = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
        self.modify_unet_forward()
        if vae_path != "default":
            self.model.vae = AutoencoderKL.from_pretrained(
                vae_path
            ).to(self.device, self.model.vae.dtype)

        self.set_lora()

        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)

        seed_everything(seed)

        self.prompt = prompt
        self.full_height = full_height
        self.full_width = full_width
        self.sup_res_h = int(0.5 * full_height)
        self.sup_res_w = int(0.5 * full_width)

        self.n_inference_step = n_inference_step
        self.n_actual_inference_step = round(inversion_strength * self.n_inference_step)
        self.guidance_scale = guidance_scale

        self.unet_feature_idx = [feature_idx]

        self.r_1 = r1
        self.r_2 = r2
        self.lam = lam
        self.beta = beta

        self.lr = latent_lr
        self.compare_mode = compare_mode

        self.t2 = drag_end_step
        self.track_per_denoise = track_per_denoise
        self.total_drag = int(track_per_denoise * self.t2)

        self.model.scheduler.set_timesteps(self.n_inference_step)

        self.do_drag = True
        self.drag_count = 0
        self.max_drag_per_track = max_drag_per_track

        self.drag_loss_threshold = drag_loss_threshold * ((2 * self.r_1) ** 2)
        self.once_drag = once_drag
        self.no_change_track_num = 0
        self.max_no_change_track_num = max_track_no_change
    

    def set_lora(self):
        if self.lora_path == "":
            print("applying default parameters")
            self.model.unet.set_default_attn_processor()
        else:
            print("applying lora: " + self.lora_path)
            self.model.unet.load_attn_procs(self.lora_path)

    def modify_unet_forward(self):
        self.model.unet.forward = override_forward(self.model.unet)

    def get_handle_target_points(self, points):
        handle_points = []
        target_points = []

        for idx, point in enumerate(points):
            cur_point = torch.tensor(
                [point[1] / self.full_height * self.sup_res_h, point[0] / self.full_width * self.sup_res_w])
            cur_point = torch.round(cur_point)
            if idx % 2 == 0:
                handle_points.append(cur_point)
            else:
                target_points.append(cur_point)
        print(f'handle points: {handle_points}')
        print(f'target points: {target_points}')
        return handle_points, target_points

    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.model.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        latents = self.model.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def forward_unet_features(self, z, t, encoder_hidden_states):
        unet_output, all_intermediate_features = self.model.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
        )

        all_return_features = []
        for idx in self.unet_feature_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (self.sup_res_h, self.sup_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)

        del all_intermediate_features
        torch.cuda.empty_cache()

        return unet_output, return_features # [1, 4, 64, 64], [1, 640, 256, 256]

    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            return_intermediates=False,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(prompt)
        else:
            text_embeddings = self.get_text_embeddings(prompt)

        latents = self.image2latent(image)

        if self.guidance_scale > 1.:
            unconditional_embeddings = self.get_text_embeddings([''] * batch_size)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("Valid timesteps: ", self.model.scheduler.timesteps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.model.scheduler.timesteps), desc="DDIM Inversion")):
            if self.n_actual_inference_step is not None and i >= self.n_actual_inference_step:
                continue

            if self.guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            t_ = self.model.scheduler.timesteps[-(i + 2)]

            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if self.guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + self.guidance_scale * (noise_pred_con - noise_pred_uncon)

            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents

    def get_original_features(self, init_code, text_embeddings):
        timesteps = self.model.scheduler.timesteps
        strat_time_step_idx = self.n_inference_step - self.n_actual_inference_step
        original_step_output = {}
        features = {}
        cur_latents = init_code.detach().clone()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps[strat_time_step_idx:],
                                       desc="Denosing for mask features")):
                if i <= self.t2:
                    model_inputs = cur_latents
                    noise_pred, F0 = self.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings)
                    cur_latents = self.model.scheduler.step(noise_pred, t, model_inputs, return_dict=False)[0]
                    original_step_output[t.item()] = cur_latents.cpu()
                    features[t.item()] = F0.cpu()

        del noise_pred, cur_latents, F0
        torch.cuda.empty_cache()

        return original_step_output, features

        # original_step_output.keys() = dict_keys([741, 721, 701, 681, 661, 641, 621, 601])
        # features.keys() = dict_keys([741, 721, 701, 681, 661, 641, 621, 601])
        # This contains all the [1,4,64,64] latents

    def get_noise_features(self, input_latents, t, text_embeddings):
        unet_output, F1 = self.forward_unet_features(input_latents, t, encoder_hidden_states=text_embeddings)
        return unet_output, F1

    #NOTE: motion supervision loss
    def cal_motion_supervision_loss(self, handle_points, target_points, F1, x_prev_updated, original_prev,
                                    interp_mask, original_features, original_points, alpha=None):
        drag_loss = 0.0
        for i_ in range(len(handle_points)):
            pi, ti = handle_points[i_], target_points[i_]
            norm_dis = (ti - pi).norm() #벡터 크기
            if norm_dis < 2.:
                continue

            di = (ti - pi) / (ti - pi).norm() * min(self.beta, norm_dis) # ANCHOR: self.beta=4, multiplies unit vector by at least 4

            original_features.requires_grad_(True)
            pi = original_points[i_]
            f0_patch = original_features[:, :, int(pi[0]) - self.r_1:int(pi[0]) + self.r_1 + 1,
                       int(pi[1]) - self.r_1:int(pi[1]) + self.r_1 + 1].detach() # self.r_1=4

            pi = handle_points[i_]
            f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_1) # f1_patch.shape=(1, 640, 9, 9) Linear interpolation due to fractional coordinates
            drag_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch) # 64*loss

        print(f'Loss from drag: {drag_loss}')
        loss = drag_loss + self.lam * ((x_prev_updated - original_prev) # self.lam=0.2
                                       * (1.0 - interp_mask)).abs().sum() 
        print('Loss total=%f' % loss)
        return loss, drag_loss

    # def cal_motion_supervision_loss_w_vector_field(self, handle_points, target_points, F1, x_prev_updated, original_prev,
    #                                    interp_mask, original_features, original_points, vector_field, vector_field_mask, 
    #                                    accelerator, optimizer, alpha=None): 

    #     # 1. Calculate vector magnitudes
    #     drag_loss = 0.0
    #     field_height, field_width, _ = vector_field.shape

    #     U = vector_field[:, :, 0]
    #     V = vector_field[:, :, 1]
    #     mask_indices = vector_field_mask.nonzero(as_tuple=True)
    #     U_masked = U[mask_indices]  # Extract values from U at mask positions
    #     V_masked = V[mask_indices]  # Extract values from V at mask positions

    #     vector_magnitudes = torch.sqrt(U_masked**2 + V_masked**2)  # Calculate vector magnitudes
    #     mean_magnitude = torch.mean(vector_magnitudes)  # Calculate mean magnitude
    #     max_magnitude = torch.max(vector_magnitudes)  # Calculate max magnitude

    #     # 2. Set parts with magnitude lower than threshold to 0
    #     # vector_magnitudes represents magnitudes of masked regions, create new mask based on this
    #     new_mask = vector_magnitudes >= (mean_magnitude + (max_magnitude - mean_magnitude)*(1/3))  # Set positions with magnitude >= threshold to True
        
    #     new_vector_field_mask = torch.zeros_like(vector_field_mask)  # Create mask filled with zeros, same size as original mask
    #     new_vector_field_mask[mask_indices] = new_mask.float()  # Set True positions in new mask to 1

    #     # 3. Get mask_indices for new mask
    #     new_mask_indices = new_vector_field_mask.nonzero(as_tuple=True)
    #     count = 0
        
    #     # Consider only points with vector magnitude above threshold, not just handle points
    #     update_interval = 2
    #     self.beta_2 = 20
    #     # step = 100
    #     for idx, (field_y, field_x) in enumerate(itertools.islice(zip(new_mask_indices[0], new_mask_indices[1]), 0, None, self.vector_step)):
    #     # for field_y, field_x in zip(new_mask_indices[0], new_mask_indices[1]):
    #         if 0 <= field_x < field_width - 10 and 0 <= field_y < field_height - 10:
    #             di = vector_field[field_y, field_x]  # di contains (x, y) vector
    #             di_norm = di.norm()
    #             di = di / di_norm * min(self.beta_2, di_norm)  # Adjust direction and magnitude based on vector field
    #             original_features.requires_grad_(True)
    #             pi = torch.tensor([field_x, field_y]).float().to(self.device)
                
    #             f0_patch = original_features[:, :, int(pi[0]) - self.r_1:int(pi[0]) + self.r_1 + 1,
    #                     int(pi[1]) - self.r_1:int(pi[1]) + self.r_1 + 1].detach()

    #             pi_new = pi + di
    #             if 0 <= pi_new[0] < field_height-self.r_1-1 and 0 <= pi_new[1] < field_width-self.r_1-1:
    #                 f1_patch = interpolate_feature_patch(F1, pi_new[0], pi_new[1], self.r_1)
    #                 try:
    #                     drag_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)
    #                     count += 1
    #                 except:
    #                     print("check")
                        
    #         # if idx % update_interval == 0:
    #         #     print(f'Intermediate Loss from drag: {drag_loss}')
    #         #     loss = drag_loss + self.lam * ((x_prev_updated - original_prev) * (1.0 - interp_mask)).abs().sum()
    #         #     accelerator.backward(loss)
    #         #     optimizer.step()
    #         #     optimizer.zero_grad()  # Reset accumulated gradients
    #         #     loss, drag_loss = 0.0, 0.0

    #     drag_loss /= count
    #     print(f'Loss from drag: {drag_loss}')
    #     loss = drag_loss + self.lam * ((x_prev_updated - original_prev)
    #                                 * (1.0 - interp_mask)).abs().sum()
    #     print('Loss total=%f' % loss)
    #     return loss, drag_loss


    def track_step(self, original_feature, original_feature_, F1, F1_, handle_points, handle_points_init):
        if self.compare_mode:
            handle_points = point_tracking(original_feature,
                                           F1, handle_points, handle_points_init, self.r_2)
        else:
            handle_points = point_tracking(original_feature_,
                                           F1_, handle_points, handle_points_init, self.r_2)

        return handle_points

    def compare_tensor_lists(self, lst1, lst2):
        if len(lst1) != len(lst2):
            return False
        return all(torch.equal(t1, t2) for t1, t2 in zip(lst1, lst2))

    def gooddrag_step(self, init_code, t, t_, text_embeddings, handle_points, target_points,
                      features, handle_points_init, original_step_output, interp_mask): #features는 original_features
        drag_latents = init_code.clone().detach() #DDIM Latents
        drag_latents.requires_grad_(True)

        first_drag = True
        need_track = False
        track_num = 0
        cur_drag_per_track = 0
        self.compare_mode = True
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision='fp16'
        )
        
        optimizer = torch.optim.Adam([drag_latents], lr=self.lr)

        drag_latents, self.model.unet, optimizer = accelerator.prepare(drag_latents, self.model.unet, optimizer)
        
        #NOTE: motion supervision & point tracking
        while track_num < self.track_per_denoise:  #self.track_per_denoise=10
            optimizer.zero_grad()
            unet_output, F1 = self.forward_unet_features(drag_latents, t, text_embeddings)
            x_prev_updated = self.model.scheduler.step(unet_output, t, drag_latents, return_dict=False)[0]

            if (need_track or first_drag) and (not self.compare_mode):
                with torch.no_grad():
                    _, F1_ = self.forward_unet_features(x_prev_updated, t_, text_embeddings)

            if first_drag:
                first_drag = False
                if self.compare_mode:
                    handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, handle_points, handle_points_init, self.r_2) #self.r_2=12
                    
                else:
                    handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, handle_points, handle_points_init, self.r_2)

                print(f'After denoise new handle points: {handle_points}, drag count: {self.drag_count}')

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                self.do_drag = False
                print('Reached the target points')
                break

            if self.no_change_track_num == self.max_no_change_track_num: # self.max_no_change_track_num=10
                self.do_drag = False
                print('Early stop.')
                break

            del unet_output
            if need_track and (not self.compare_mode):
                del _
            torch.cuda.empty_cache()
            #NOTE: motion supervision loss
            loss, drag_loss = self.cal_motion_supervision_loss(handle_points, target_points, F1, x_prev_updated, 
                                                               original_step_output[t.item()].cuda(), interp_mask,
                                                               original_features=features[t.item()].cuda(),
                                                               original_points=handle_points_init)
            
            accelerator.backward(loss)

            optimizer.step()

            cur_drag_per_track += 1
            need_track = (cur_drag_per_track == self.max_drag_per_track) or ( # self.max_drag_per_track=3
                    drag_loss <= self.drag_loss_threshold) or self.once_drag # self.drag_loss_threshold=0
            if need_track:
                track_num += 1
                handle_points_cur = copy.deepcopy(handle_points)
                if self.compare_mode:
                    handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, handle_points, handle_points_init, self.r_2)
                
                else:
                    handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, handle_points, handle_points_init, self.r_2)

                if self.compare_tensor_lists(handle_points, handle_points_cur):
                    self.no_change_track_num += 1
                    print(f'{self.no_change_track_num} times handle points no changes.')
                else:
                    self.no_change_track_num = 0

                self.drag_count += 1
                cur_drag_per_track = 0
                print(f'New handle points: {handle_points}, drag count: {self.drag_count}')

        init_code = drag_latents.clone().detach()
        init_code.requires_grad_(False)
        del optimizer, drag_latents
        torch.cuda.empty_cache()

        return init_code, handle_points

    def flowdrag_step(self, init_code, t, t_, text_embeddings, handle_points, target_points,
                      features, handle_points_init, original_step_output, interp_mask,
                      num_user_points=1):
        """
        Optimized FlowDrag step with adaptive point selection and hierarchical supervision.

        Args:
            num_user_points: Number of user-provided points (rest are from vector field sampling)
        """
        drag_latents = init_code.clone().detach()
        drag_latents.requires_grad_(True)

        first_drag = True
        need_track = False
        track_num = 0
        cur_drag_per_track = 0
        self.compare_mode = True
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision='fp16'
        )

        optimizer = torch.optim.Adam([drag_latents], lr=self.lr)
        drag_latents, self.model.unet, optimizer = accelerator.prepare(drag_latents, self.model.unet, optimizer)

        # Separate user points and vector field points
        user_handle_points = handle_points[:num_user_points]
        user_target_points = target_points[:num_user_points]
        user_handle_points_init = handle_points_init[:num_user_points]

        vf_handle_points = handle_points[num_user_points:]
        vf_target_points = target_points[num_user_points:]
        vf_handle_points_init = handle_points_init[num_user_points:]

        # Active points tracking: remove points that reached target
        active_vf_indices = list(range(len(vf_handle_points)))

        #NOTE: motion supervision & point tracking
        while track_num < self.track_per_denoise:  #self.track_per_denoise=10
            optimizer.zero_grad()
            unet_output, F1 = self.forward_unet_features(drag_latents, t, text_embeddings)
            x_prev_updated = self.model.scheduler.step(unet_output, t, drag_latents, return_dict=False)[0]

            if (need_track or first_drag) and (not self.compare_mode):
                with torch.no_grad():
                    _, F1_ = self.forward_unet_features(x_prev_updated, t_, text_embeddings)

            if first_drag:
                first_drag = False
                # Track user points
                if self.compare_mode:
                    user_handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, user_handle_points, user_handle_points_init, self.r_2)
                else:
                    user_handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, user_handle_points, user_handle_points_init, self.r_2)

                # Track only active vector field points
                if len(active_vf_indices) > 0:
                    active_vf_handle = [vf_handle_points[i] for i in active_vf_indices]
                    active_vf_init = [vf_handle_points_init[i] for i in active_vf_indices]
                    if self.compare_mode:
                        active_vf_handle = point_tracking(features[t.item()].cuda(),
                                                       F1, active_vf_handle, active_vf_init, self.r_2)
                    else:
                        active_vf_handle = point_tracking(features[t_.item()].cuda(),
                                                       F1_, active_vf_handle, active_vf_init, self.r_2)
                    # Update back to main list
                    for idx, i in enumerate(active_vf_indices):
                        vf_handle_points[i] = active_vf_handle[idx]

                # Combine for printing
                all_handle_points = user_handle_points + vf_handle_points
                print(f'After denoise: user points={len(user_handle_points)}, vf points={len(active_vf_indices)}, drag count={self.drag_count}')

            # Check if user points reached targets (higher priority)
            if check_handle_reach_target(user_handle_points, user_target_points):
                self.do_drag = False
                print('User points reached targets')
                break

            if self.no_change_track_num == self.max_no_change_track_num:
                self.do_drag = False
                print('Early stop.')
                break

            del unet_output
            if need_track and (not self.compare_mode):
                del _
            torch.cuda.empty_cache()

            #NOTE: Hierarchical motion supervision loss
            # 1. User points with high weight
            user_loss = 0.0
            for i_ in range(len(user_handle_points)):
                pi, ti = user_handle_points[i_], user_target_points[i_]
                norm_dis = (ti - pi).norm()
                if norm_dis < 2.:
                    continue

                di = (ti - pi) / norm_dis * min(self.beta, norm_dis)
                pi_orig = user_handle_points_init[i_]
                f0_patch = features[t.item()].cuda()[:, :, int(pi_orig[0]) - self.r_1:int(pi_orig[0]) + self.r_1 + 1,
                           int(pi_orig[1]) - self.r_1:int(pi_orig[1]) + self.r_1 + 1].detach()

                f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_1)
                user_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # Apply high weight to user points (3x)
            drag_loss = 3.0 * user_loss if len(user_handle_points) > 0 else 0.0

            # 2. Vector field points with adaptive selection (use top-K based on distance)
            if len(active_vf_indices) > 0:
                # Calculate distance to target for each active VF point
                vf_distances = []
                for i in active_vf_indices:
                    dist = (vf_handle_points[i] - vf_target_points[i]).norm()
                    vf_distances.append((i, dist.item()))

                # Sort by distance (larger distance = more important)
                vf_distances.sort(key=lambda x: x[1], reverse=True)

                # Use top-K points (adaptive: min 5, max 10)
                max_vf_points = min(10, max(5, len(active_vf_indices) // 2))
                selected_vf_indices = [idx for idx, _ in vf_distances[:max_vf_points]]

                vf_loss = 0.0
                vf_count = 0
                for i in selected_vf_indices:
                    pi, ti = vf_handle_points[i], vf_target_points[i]
                    norm_dis = (ti - pi).norm()
                    if norm_dis < 2.:
                        # Remove from active list if reached target
                        if i in active_vf_indices:
                            active_vf_indices.remove(i)
                        continue

                    di = (ti - pi) / norm_dis * min(self.beta, norm_dis)
                    pi_orig = vf_handle_points_init[i]
                    f0_patch = features[t.item()].cuda()[:, :, int(pi_orig[0]) - self.r_1:int(pi_orig[0]) + self.r_1 + 1,
                               int(pi_orig[1]) - self.r_1:int(pi_orig[1]) + self.r_1 + 1].detach()

                    f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_1)
                    vf_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)
                    vf_count += 1

                # Add VF loss with lower weight
                if vf_count > 0:
                    drag_loss += vf_loss / vf_count

            print(f'Loss from drag (user: {len(user_handle_points)}, active vf: {len(active_vf_indices)}): {drag_loss}')

            loss = drag_loss + self.lam * ((x_prev_updated - original_step_output[t.item()].cuda())
                                        * (1.0 - interp_mask)).abs().sum()
            print('Loss total=%f' % loss)

            accelerator.backward(loss)
            optimizer.step()

            cur_drag_per_track += 1
            need_track = (cur_drag_per_track == self.max_drag_per_track) or (
                    drag_loss <= self.drag_loss_threshold) or self.once_drag
            if need_track:
                track_num += 1
                user_handle_points_cur = copy.deepcopy(user_handle_points)

                # Track user points
                if self.compare_mode:
                    user_handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, user_handle_points, user_handle_points_init, self.r_2)
                else:
                    user_handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, user_handle_points, user_handle_points_init, self.r_2)

                # Track active VF points
                if len(active_vf_indices) > 0:
                    active_vf_handle = [vf_handle_points[i] for i in active_vf_indices]
                    active_vf_init = [vf_handle_points_init[i] for i in active_vf_indices]
                    if self.compare_mode:
                        active_vf_handle = point_tracking(features[t.item()].cuda(),
                                                       F1, active_vf_handle, active_vf_init, self.r_2)
                    else:
                        active_vf_handle = point_tracking(features[t_.item()].cuda(),
                                                       F1_, active_vf_handle, active_vf_init, self.r_2)
                    for idx, i in enumerate(active_vf_indices):
                        vf_handle_points[i] = active_vf_handle[idx]

                if self.compare_tensor_lists(user_handle_points, user_handle_points_cur):
                    self.no_change_track_num += 1
                    print(f'{self.no_change_track_num} times handle points no changes.')
                else:
                    self.no_change_track_num = 0

                self.drag_count += 1
                cur_drag_per_track = 0
                print(f'Track {track_num}: user points={len(user_handle_points)}, active vf={len(active_vf_indices)}')

        init_code = drag_latents.clone().detach()
        init_code.requires_grad_(False)
        del optimizer, drag_latents
        torch.cuda.empty_cache()

        # Return only user handle points for consistency with baseline
        return init_code, user_handle_points

    def prepare_mask(self, mask):
        mask = torch.from_numpy(mask).float() / 255.
        mask[mask > 0.0] = 1.0
        mask = rearrange(mask, "h w -> 1 1 h w").cuda()
        mask = F.interpolate(mask, (self.sup_res_h, self.sup_res_w), mode="nearest")
        return mask
        
    def set_latent_masactrl(self):
        editor = MutualSelfAttentionControl(start_step=0,
                                            start_layer=10,
                                            total_steps=self.n_inference_step,
                                            guidance_scale=self.guidance_scale)
        if self.lora_path == "":
            register_attention_editor_diffusers(self.model, editor, attn_processor='attn_proc')
        else:
            register_attention_editor_diffusers(self.model, editor, attn_processor='lora_attn_proc')

    def get_intermediate_images(self, intermediate_images, intermediate_images_original, intermediate_images_t_idx,
                                valid_timestep, text_embeddings):
        for i in range(len(intermediate_images)-1):
            current_original_code = intermediate_images_original[i].to(self.device)
            current_init_code = intermediate_images[i].to(self.device)

            self.set_latent_masactrl()

            for inter_i, inter_t in enumerate(valid_timestep[intermediate_images_t_idx[i] + 1:]):
                with torch.no_grad():
                    noise_pred_all = self.model.unet(torch.cat([current_original_code, current_init_code]), inter_t,
                                                     encoder_hidden_states=torch.cat(
                                                         [text_embeddings, text_embeddings]))
                    noise_pred = noise_pred_all[1]
                    noise_pred_original = noise_pred_all[0]
                    current_init_code = \
                        self.model.scheduler.step(noise_pred, inter_t, current_init_code, return_dict=False)[0]
                    current_original_code = \
                        self.model.scheduler.step(noise_pred_original, inter_t, current_original_code,
                                                  return_dict=False)[0]
            intermediate_images[i] = self.latent2image(current_init_code, return_type="pt").cpu()
        intermediate_images.pop()
        return intermediate_images

    def good_drag(self,
                  source_image,
                  points,
                  mask,
                  return_intermediate_images=False,
                  return_intermediate_features=False,
                  ):
        #NOTE: DDIM Inversion
        init_code = self.invert(source_image, self.prompt)  # [1, 4, 64, 64]
        original_init = init_code.detach().clone()
        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(self.prompt)
            text_embeddings = text_embeddings.detach()
        else:
            text_embeddings = self.get_text_embeddings(self.prompt).detach()

        self.model.text_encoder.to('cpu')
        self.model.vae.encoder.to('cpu')

        timesteps = self.model.scheduler.timesteps
        start_time_step_idx = self.n_inference_step - self.n_actual_inference_step

        handle_points, target_points = self.get_handle_target_points(points)
        original_step_output, features = self.get_original_features(init_code, text_embeddings)

        handle_points_init = copy.deepcopy(handle_points)
        mask = self.prepare_mask(mask)
        interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')

        intermediate_features = [init_code.detach().clone().cpu()] if return_intermediate_features else []
        valid_timestep = timesteps[start_time_step_idx:] #start_time_step_idx=12  
        set_mutual = True

        intermediate_images, intermediate_images_original, intermediate_images_t_idx = [], [], []

        did_drag = False
        
        #NOTE: denoising and drag => valid_timestep 741, 721... # self.t2 = 7
        for i, t in enumerate(tqdm(valid_timestep,
                                   desc="Drag and Denoise")):
            if i < self.t2 and self.do_drag and (self.no_change_track_num != self.max_no_change_track_num): #self.t2=7, self.max_no_change_track_num=10
                t_ = valid_timestep[i + 1]
                init_code, handle_points = self.gooddrag_step(init_code, t, t_, text_embeddings, handle_points,
                                                              target_points, features, handle_points_init, #features는 original_features
                                                              original_step_output, interp_mask)
                
                did_drag = True
            else:
                if set_mutual:
                    set_mutual = False
                    self.set_latent_masactrl()

            with torch.no_grad():
                noise_pred_all = self.model.unet(torch.cat([original_init, init_code]), t,
                                                 encoder_hidden_states=torch.cat([text_embeddings, text_embeddings]))
                noise_pred = noise_pred_all[1]
                noise_pred_original = noise_pred_all[0]
                init_code = self.model.scheduler.step(noise_pred, t, init_code, return_dict=False)[0]
                original_init = self.model.scheduler.step(noise_pred_original, t, original_init, return_dict=False)[0]

            if did_drag and return_intermediate_images:
                current_init_code = init_code.detach().clone()
                current_original_code = original_init.detach().clone()

                intermediate_images.append(current_init_code.cpu())
                intermediate_images_original.append(current_original_code.cpu())
                intermediate_images_t_idx.append(i)
            did_drag = False
            if return_intermediate_features:
                intermediate_features.append(init_code.detach().clone().cpu())

        if return_intermediate_images:
            intermediate_images = self.get_intermediate_images(intermediate_images, intermediate_images_original,
                                                               intermediate_images_t_idx, valid_timestep, text_embeddings)

        image = self.latent2image(init_code, return_type="pt")
        return image, intermediate_features, handle_points, intermediate_images
    

    def flow_drag(self,
                  source_image,
                  points,
                  mask,
                  return_intermediate_images=False,
                  return_intermediate_features=False,
                  num_user_points=1, 
                  ):
        """
        FlowDrag with optimized hierarchical point processing.

        Args:
            num_user_points: Number of user-provided points (first N points in the list)
                            Rest are vector field sampled points
        """
        init_code = self.invert(source_image, self.prompt) #NOTE: DDIM Inversion
        original_init = init_code.detach().clone()
        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(self.prompt)
            text_embeddings = text_embeddings.detach()
        else:
            text_embeddings = self.get_text_embeddings(self.prompt).detach()

        self.model.text_encoder.to('cpu')
        self.model.vae.encoder.to('cpu')

        timesteps = self.model.scheduler.timesteps
        start_time_step_idx = self.n_inference_step - self.n_actual_inference_step

        handle_points, target_points = self.get_handle_target_points(points)
        original_step_output, features = self.get_original_features(init_code, text_embeddings)

        mask = self.prepare_mask(mask)
        interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')

        print(f"FlowDrag - Total points: {len(handle_points)}, User points: {num_user_points}, VF points: {len(handle_points) - num_user_points}")
        print(f"Handle points: {handle_points[:num_user_points]} (user) + {len(handle_points) - num_user_points} VF points")

        handle_points_init = copy.deepcopy(handle_points)

        intermediate_features = [init_code.detach().clone().cpu()] if return_intermediate_features else []
        valid_timestep = timesteps[start_time_step_idx:]
        set_mutual = True

        intermediate_images, intermediate_images_original, intermediate_images_t_idx = [], [], []

        did_drag = False

        #NOTE: denoising and drag => valid_timestep 741, 721... 
        for i, t in enumerate(tqdm(valid_timestep,
                                   desc="FlowDrag: Drag and Denoise")):
            if i < self.t2 and self.do_drag and (self.no_change_track_num != self.max_no_change_track_num): #self.t2=7
                t_ = valid_timestep[i + 1]
                init_code, handle_points = self.flowdrag_step(init_code, t, t_, text_embeddings, handle_points,
                                                              target_points, features, handle_points_init,
                                                              original_step_output, interp_mask,
                                                              num_user_points=num_user_points)

                did_drag = True
            else:
                if set_mutual:
                    set_mutual = False
                    self.set_latent_masactrl()

            with torch.no_grad():
                noise_pred_all = self.model.unet(torch.cat([original_init, init_code]), t,
                                                 encoder_hidden_states=torch.cat([text_embeddings, text_embeddings]))
                noise_pred = noise_pred_all[1]
                noise_pred_original = noise_pred_all[0]
                init_code = self.model.scheduler.step(noise_pred, t, init_code, return_dict=False)[0]
                original_init = self.model.scheduler.step(noise_pred_original, t, original_init, return_dict=False)[0]

            if did_drag and return_intermediate_images:
                current_init_code = init_code.detach().clone()
                current_original_code = original_init.detach().clone()

                intermediate_images.append(current_init_code.cpu())
                intermediate_images_original.append(current_original_code.cpu())
                intermediate_images_t_idx.append(i)
            did_drag = False
            if return_intermediate_features:
                intermediate_features.append(init_code.detach().clone().cpu())

        if return_intermediate_images:
            intermediate_images = self.get_intermediate_images(intermediate_images, intermediate_images_original,
                                                               intermediate_images_t_idx, valid_timestep, text_embeddings)

        image = self.latent2image(init_code, return_type="pt")
        # Return only user handle points for consistency
        return image, intermediate_features, handle_points, intermediate_images
