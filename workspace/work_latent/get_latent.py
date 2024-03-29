import PIL
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils import load_image, export_to_video

from svd_class.original import OrgStableVideoDiffusionPipeline

from typing import Callable, Dict, List, Optional, Union

import torch
import os
import time
import json
import shutil

image_path = "/home/nas2_userG/junhahyung/kkn/LucidDreamer/local/image/dreaming/custom/0_image.png"
base_image = load_image(image_path)
generator = torch.manual_seed(42)
org_pipe = OrgStableVideoDiffusionPipeline.from_pretrained(
    "/home/nas2_userG/junhahyung/kkn/checkpoint/stable-video-diffusion", torch_dtype=torch.float32
)
org_pipe.enable_sequential_cpu_offload()

file_name = "org"
frame_dir = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/workspace/work_latent/frames/{file_name}"
latent_dir = f"/home/nas2_userG/junhahyung/kkn/drag-SVD/workspace/work_latent/latents/{file_name}"

os.makedirs(frame_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)

num_frames = 4
do_classifier_free_guidance = True

frames = org_pipe(base_image,
                latent_dir=latent_dir,
                num_frames=num_frames,
                generator=generator,
                motion_bucket_id=127,
                fps=7,
                ).frames[0]

for i, frame in enumerate(frames):
    frame.save(os.path.join(frame_dir, f"frame_{i}.png"))