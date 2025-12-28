from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
import torch, PIL.Image as Image
from tqdm import tqdm
import numpy as np
from pathlib import Path

# created by kookie 2025-07-08 19:13:46

SD_15_PATH = "/mnt/hdd/sunjaeyoon/workspace/pretrain_SD_models/CompVis/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    SD_15_PATH,
    torch_dtype=torch.float16,
).to("cuda")

ddim = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                     beta_schedule="scaled_linear", clip_sample=False,
                     set_alpha_to_one=False, steps_offset=1)
pipe.scheduler = ddim

# Set number of inference steps
n_inference_steps = 50
guidance_scale = 1.0
inversion_strength = 0.75  # pipeline.py의 기본값
pipe.scheduler.set_timesteps(n_inference_steps)

# Load and prepare image
# init = Image.open("/mnt/hdd/sunjaeyoon/workspace/ICML2025/FlowDrag/samples/penguin/penguin_.jpg")
# init = Image.open("/mnt/hdd/sunjaeyoon/workspace/ICML2025/FlowDrag/small_dog.png")
init = Image.open("/mnt/hdd/sunjaeyoon/workspace/ICML2025/FlowDrag/small_dog_projection.png")
save_name = Path(init.filename).stem

# resize 512x512 and convert to RGB (remove alpha channel if exists)
init = init.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB')

@torch.no_grad()
def image2latent(image, vae):
    """Pipeline.py 방식의 image2latent 함수"""
    if type(image) is Image.Image:
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float16)
    
    latents = vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents

@torch.no_grad()
def get_text_embeddings(prompt, tokenizer, text_encoder):
    """Pipeline.py 방식의 text embedding 함수"""
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]
    return text_embeddings

def inv_step(model_output, timestep, x, scheduler):
    """
    Pipeline.py의 inv_step 함수와 동일
    """
    next_step = timestep
    timestep = min(timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999)
    alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
    x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
    return x_next, pred_x0

# 이미지를 latent로 변환
latents = image2latent(init, pipe.vae)

# Text embeddings 준비
prompt = ""
text_embeddings = get_text_embeddings(prompt, pipe.tokenizer, pipe.text_encoder)

# Guidance scale 처리 (pipeline.py 방식)
if guidance_scale > 1.:
    unconditional_embeddings = get_text_embeddings([''], pipe.tokenizer, pipe.text_encoder)
    text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

# Pipeline.py 방식의 실제 inversion 스텝 계산
n_actual_inference_step = round(inversion_strength * n_inference_steps)
print(f"n_actual_inference_step: {n_actual_inference_step}")

timesteps = pipe.scheduler.timesteps
print("Valid timesteps: ", timesteps)

current_latents = latents.clone()
latents_list = [current_latents.clone()]

# DDIM Inversion (Pipeline.py 방식)
with torch.no_grad():
    for i, t in enumerate(tqdm(reversed(timesteps), desc="DDIM Inversion")):
        if n_actual_inference_step is not None and i >= n_actual_inference_step:
            continue
            
        if guidance_scale > 1.:
            model_inputs = torch.cat([current_latents] * 2)
        else:
            model_inputs = current_latents

        noise_pred = pipe.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
        
        if guidance_scale > 1.:
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        
        current_latents, pred_x0 = inv_step(noise_pred, t, current_latents, pipe.scheduler)
        latents_list.append(current_latents.clone())

# Get the latents after inversion
inverted_latents = current_latents

# Decode the inverted latents back to image
with torch.no_grad():
    inverted_latents_decoded = 1 / 0.18215 * inverted_latents
    noisy_img = pipe.vae.decode(inverted_latents_decoded)['sample']
    noisy_img = (noisy_img / 2 + 0.5).clamp(0, 1)
    noisy_img = noisy_img.cpu().permute(0, 2, 3, 1).numpy()[0]
    noisy_img = (noisy_img * 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_img)

noisy_img.save(f"{save_name}_ddim_step{n_actual_inference_step}.png")
print(f"Saved image after {n_actual_inference_step} steps of DDIM inversion")
print(f"inversion_strength: {inversion_strength}")