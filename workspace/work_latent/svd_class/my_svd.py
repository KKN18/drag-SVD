import PIL
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils import load_image, export_to_video

from typing import Callable, Dict, List, Optional, Union

import torch
import os
import time
import json


class MyStableVideoDiffusionPipeline(StableVideoDiffusionPipeline):
    def save_latents(self, latents, save_dir):
        torch.save(latents, os.path.join(save_dir, f"latents.pt"))
        
        print("Latents saved")
        return
    
    @torch.no_grad()
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        latent_dir: str,
        num_frames: Optional[int] = 8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        org_latents: Optional[torch.Tensor] = None,
    ):
        num_inference_steps = 25
        min_guidance_scale = 1.0
        max_guidance_scale = 3.0
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.02
        decode_chunk_size = 2
        num_videos_per_prompt = 1
        output_type = "pil"
        callback_on_step_end = None
        callback_on_step_end_tensor_inputs = ["latents"]
        return_dict = True
        
        
        batch_size = 1
        self._guidance_scale = max_guidance_scale
        device = self._execution_device
        
        fps = fps - 1
        
        image = images[0].copy()
        
        # 0. Default height and width to unet
        height, width = image.size
        
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        vae_latent = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        vae_latent = vae_latent.to(image_embeddings.dtype)
        vae_latent = vae_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1) # [1, 8, 4, 64, 64]

        # 각 이미지들을 vae encoding해서 latents 추출
        latents = []
        for image in images:
            image = self.image_processor.preprocess(image, height=height, width=width).to(device)
            noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
            
            latent = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=False,
            ) # [1, 4, 64, 64]
            print(f"latent shape: {latent.shape}")
            latent = latent.to(image_embeddings.dtype)
            latent = latent.unsqueeze(1)
            latents.append(latent)
        latents = torch.cat(latents, dim=1) # [1, 8, 4, 64, 64]
        print(f"latents shape: {latents.shape}")
        latents *= self.vae.config.scaling_factor
        
        self.save_latents(latents, latent_dir)
        
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        
        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)