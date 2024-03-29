import PIL
from PIL import Image

from diffusers import StableVideoDiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils import load_image, export_to_video
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims, _resize_with_antialiasing

from typing import Callable, Dict, List, Optional, Union

import torch
import os
import time
import json
import shutil
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from diffusers.utils import BaseOutput
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from types import SimpleNamespace
from einops import rearrange
from copy import deepcopy

from model import DragSVDPipeline
from util.utils import diffusion_update

device = torch.device("cuda")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                        beta_schedule="scaled_linear", clip_sample=False,
                        set_alpha_to_one=False, steps_offset=1)
model = DragSVDPipeline.from_pretrained("/home/nas2_userG/junhahyung/kkn/checkpoint/stable-video-diffusion",
                                        scheduler=scheduler, torch_dtype=torch.float32, safety_checker=None)
model.modify_unet_forward()
model.scheduler = scheduler
model.enable_sequential_cpu_offload()

seed = 42

def preprocess_image(image,
                     device,
                     dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image

#---------------Option----------------#
inversion_strength = 0.9 # TODO: 0.7 / max: 0.35
inference_step = 25 # TODO: 50
guidance_scale = 1.0

infer_do_classifier_free_guidance = True
min_guidance_scale = 1.0 # infer
max_guidance_scale = 3.0 # infer


lr = 0.01
lam = 0.1 # regularization strength on unmasked areas
n_pix_step = 0 # number of gradient descent
num_frames = 4

alias = "main"
pixel_type = "mvright_pixels"

image_path = "/home/nas2_userG/junhahyung/kkn/drag-SVD/image/source_image.png"
latent_dir_path = "/home/nas2_userG/junhahyung/kkn/drag-SVD/latents"
mask_path = "/home/nas2_userG/junhahyung/kkn/drag-SVD/image/mask.png"
target_dir_path = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/{pixel_type}"
base_dir = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/output/{alias}"
#-------------------------------------#
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    
base_dir = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/output/{alias}"
folder_count = sum(os.path.isdir(os.path.join(base_dir, name)) for name in os.listdir(base_dir))
output_dir_path = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/output/{alias}/#{folder_count}_{pixel_type}_inferstep_{inference_step}_numframes_{num_frames}_npixstep_{n_pix_step}_inversionstrength_{inversion_strength}"

svd_image = load_image(image_path)

source_image = Image.open(image_path)
source_image = np.array(source_image)

mask = Image.open(mask_path)
mask = np.array(mask)

# vae_latent = torch.load(f"{latent_dir_path}/vae_latent.pt")
# print(f"vae_latent.shape: {vae_latent.shape}")
# image_embedding = torch.load(f"{latent_dir_path}/image_embedding.pt")
# print(f"image_embedding.shape: {image_embedding.shape}")
latents = torch.load(f"{latent_dir_path}/4_latents.pt")
print(f"latents.shape: {latents.shape}")

args = SimpleNamespace()
args.image = source_image
args.n_inference_step = inference_step
args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
args.guidance_scale = guidance_scale

args.unet_feature_idx= [3]

args.lr = lr
args.n_pix_step = n_pix_step
args.r_m = 1
args.r_p = 3
args.lam = lam

full_h, full_w = source_image.shape[:2]
args.sup_res_h = int(0.5*full_h)
args.sup_res_w = int(0.5*full_w)

args.num_frames = num_frames

source_image = preprocess_image(source_image, 
                                device,
                                dtype=torch.float32)

mask = torch.from_numpy(mask).float() / 255.
mask[mask > 0.0] = 1.0
mask = rearrange(mask, "h w -> 1 1 h w").cuda() # [1, 1, 512, 512]
mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest") # [1, 1, 256, 256]

handle_points_all = []
target_points_all = []
for i in range(1, num_frames+1): 
    file_path = os.path.join(target_dir_path, f'target_0_{i}.npy')
    
    # 파일이 존재하면 처리
    if os.path.exists(file_path):
        points = np.load(file_path)

        handle_points = []
        target_points = []
        
        # 각 픽셀 좌표에 대한 처리
        for point in points:
            # 현재 점을 변환
            cur_handle_point = torch.tensor([point[0]/full_w*args.sup_res_w, point[1]/full_h*args.sup_res_h], dtype=torch.float32)
            cur_target_point = torch.tensor([point[2]/full_w*args.sup_res_w, point[3]/full_h*args.sup_res_h], dtype=torch.float32)
            
            # 반올림
            cur_handle_point = torch.round(cur_handle_point)
            cur_target_point = torch.round(cur_target_point)

            # 리스트에 추가
            handle_points.append(cur_handle_point)
            target_points.append(cur_target_point)

        # 전체 리스트에 추가
        handle_points_all.append(torch.stack(handle_points))  # torch.stack을 사용하여 차원을 추가
        target_points_all.append(torch.stack(target_points))  # torch.stack을 사용하여 차원을 추가

# 결과 출력
print('handle points:', handle_points_all)
print('target points:', target_points_all)

# TODO: Implement LoRA
# if lora_path == "":
#     print("applying default parameters")
#     model.unet.set_default_attn_processor()
# else:
#     print("applying lora: " + lora_path)
#     model.unet.load_attn_procs(lora_path)

if guidance_scale > 1:
    do_classifier_free_guidance = True
else:
    do_classifier_free_guidance = False
    
vae_latent = model.image2latent(source_image,
                                do_classifier_free_guidance=do_classifier_free_guidance,)
    
image_embeddings = model.get_image_embeddings(svd_image,
                                             do_classifier_free_guidance=do_classifier_free_guidance,)
print(f"image_embeddings.shape: {image_embeddings.shape}")
latents = latents[:, :num_frames]

invert_code = model.invert(svd_image,
                           latents,
                           num_frames=num_frames,
                           encoder_hidden_states=image_embeddings,
                           guidance_scale=args.guidance_scale,
                           num_inference_steps=args.n_inference_step,
                           num_actual_inference_steps=args.n_actual_inference_step,)

# gen_frames = model(
#     svd_image,
#     encoder_hidden_states=image_embeddings,
#     batch_size=1,
#     num_frames=num_frames,
#     latents=invert_code,
#     guidance_scale=args.guidance_scale,
#     num_inference_steps=args.n_inference_step,
#     num_actual_inference_steps=args.n_actual_inference_step,
# )[0]

# os.makedirs(output_dir_path, exist_ok=True)

# for frame_idx, frame in enumerate(gen_frames):
#     frame.save(os.path.join(output_dir_path, f"init_{frame_idx}.png"))

# print("Init saved")

torch.cuda.empty_cache()

init_code = invert_code
init_code_orig = deepcopy(init_code)
model.scheduler.set_timesteps(args.n_inference_step) 
t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step] # 50 - 35 = 15
print(f"update timestep: {t}")
# init_code = init_code.float()
# image_embeddings = image_embeddings.float()
# model.unet = model.unet.float()

updated_init_code = diffusion_update(
    model,
    init_code,
    vae_latent,
    image_embeddings,
    t,
    handle_points_all,
    target_points_all,
    mask,
    args
)

#TODO: use float16
# updated_init_code = updated_init_code.half()
# text_embeddings = text_embeddings.half()
# model.unet = model.unet.half()

# empty cache to save memory
torch.cuda.empty_cache()

gen_frames = model(
    svd_image,
    encoder_hidden_states=image_embeddings,
    batch_size=1,
    num_frames=num_frames,
    latents=updated_init_code,
    do_classifier_free_guidance=infer_do_classifier_free_guidance,
    min_guidance_scale=min_guidance_scale,
    max_guidance_scale=max_guidance_scale,
    num_inference_steps=args.n_inference_step,
    num_actual_inference_steps=args.n_actual_inference_step,
)[0]

os.makedirs(output_dir_path, exist_ok=True)

for frame_idx, frame in enumerate(gen_frames):
    frame.save(os.path.join(output_dir_path, f"frame_{frame_idx}.png"))

print(f"Frame saved in {output_dir_path}")