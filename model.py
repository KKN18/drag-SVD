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

@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None

def override_forward(self):
    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_intermediates: bool = False,
    ):
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
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples
                

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        all_intermediate_features = [sample]

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
            all_intermediate_features.append(sample)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward
    
# scaling factor: self.vae.config.scaling_factor
class DragSVDPipeline(StableVideoDiffusionPipeline):
    # must call this function when initialize
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)
        
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0
    
    @torch.no_grad()
    def image2latent(self,
                    image,
                    height=512,
                    width=512,
                    noise_aug_strength=0.02,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=True,
                    ):
        image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        
        noise = randn_tensor(image.shape, device=torch.device("cuda"), dtype=torch.float32)
        image = image + noise_aug_strength * noise
        
        image = image.to("cuda")
        image_latents = self.vae.encode(image).latent_dist.mode()
        
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])
        
        return image_latents
    
    @torch.no_grad()
    def latent2video(self, latents, num_frames=8, decode_chunk_size=8):
        frames = self.decode_latents(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        return frames
    
    def latent2video_grad(self, latents, num_frames=8, decode_chunk_size=8):
        frames = self.decode_latents(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        return frames
    
    @torch.no_grad()
    def get_image_embeddings(self, image, num_videos_per_prompt=1, do_classifier_free_guidance=True):
        dtype = next(self.image_encoder.parameters()).dtype
        device = torch.device("cuda")

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings
    
    @torch.no_grad()
    def get_neg_embeddings(self, image):
        image_embeddings = self.get_image_embeddings(image, do_classifier_free_guidance=True)
        total_length = image_embeddings.size(0)
        n = total_length // 2
        neg_embedding = image_embeddings[:n]
        return neg_embedding
        
    
    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(
        self,
        latent_model_input,
        t,
        encoder_hidden_states,
        added_time_ids,
        layer_idx=[0],
        interp_res_h=256,
        interp_res_w=256):
        unet_output, all_intermediate_features = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_intermediates=True
            )

        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)
        return unet_output, return_features
    
    @torch.no_grad()
    def __call__(
        self,
        image,
        encoder_hidden_states=None,
        num_frames=8,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        do_classifier_free_guidance=True,
        min_guidance_scale=1.0,
        max_guidance_scale=3.0,
        latents=None,
        neg_image = None,
        return_intermediates=False,
        decode_chunk_size=2,
        output_type="pil"
    ):
        device = torch.device("cuda")
        
        added_time_ids = self._get_add_time_ids(
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            dtype=torch.float32,
            batch_size=1,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        
        encoder_hidden_states = self.get_image_embeddings(image, 
                                                          do_classifier_free_guidance=do_classifier_free_guidance)
        vae_latent = self.image2latent(image, 
                                       height=height, 
                                       width=width, 
                                       num_videos_per_prompt=1, 
                                       do_classifier_free_guidance=do_classifier_free_guidance)
        vae_latent = vae_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * 1, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        
        self.scheduler.set_timesteps(num_inference_steps)
        if return_intermediates:
            latents_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            print(f"i: {i}, t: {t}")
            
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            latent_model_input = torch.cat([latent_model_input, vae_latent], dim=2)    
            # print(f"latent_model_input.shape: {latent_model_input.shape}")
            # predict the noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
            )
            
            # TODO: guidance_scale이 frame마다 다르게 적용되어야 함
            if do_classifier_free_guidance:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                
            # compute the previous noise sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if return_intermediates:
                latents_list.append(latents)
        
        frames = self.latent2video(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        if return_intermediates:
            return frames, latents_list
        return frames
    
    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        latents: torch.Tensor,
        num_frames=8,
        encoder_hidden_states=None,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = 1
        
        do_classifier_free_guidance = True if guidance_scale > 1 else False
        
        added_time_ids = self._get_add_time_ids(
            fps=num_frames - 1,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            dtype=torch.float32,
            batch_size=1,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        
        encoder_hidden_states = self.get_image_embeddings(image, 
                                                          do_classifier_free_guidance=do_classifier_free_guidance)
        
        vae_latent = self.image2latent(image, 
                                        height=512, 
                                        width=512, 
                                        num_videos_per_prompt=1, 
                                        do_classifier_free_guidance=do_classifier_free_guidance)

        vae_latent = vae_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
            
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue
            print(f"i: {i}, t: {t}")
            if guidance_scale > 1.:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            latent_model_input = torch.cat([latent_model_input, vae_latent], dim=2)
            # predict the noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
            ) # [1, 4, 4, 64, 64]
            if guidance_scale > 1:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            # pred_x0_list.append(pred_x0)
        
        if return_intermediates:
            return latents, latents_list
        return latents