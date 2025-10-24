import os
import numpy as np
import torch
import logging
import sys
from time import time
from PIL import Image
import argparse
from diffusers import FluxPipeline
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from module_init.modules import flux_attn_call2_0, FluxTransformer2DModelForward, FluxTransformerBlockForward

def update_transformer_methods(model):
    """Update forward methods for transformer components."""
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        model.forward = FluxTransformer2DModelForward.__get__(model, model.__class__)
    for name, component in model.named_children():
        if component.__class__.__name__ == 'FluxTransformerBlock':
            component.forward = FluxTransformerBlockForward.__get__(component, component.__class__)
        update_transformer_methods(component)
    return model

def initialize_pipeline(pipeline, use_8bit=False):
    """Configure pipeline for transformer processing."""
    if 'transformer' in vars(pipeline) and pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
        FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
        pipeline.transformer = update_transformer_methods(pipeline.transformer)
    return pipeline

def calculate_similarity(vec1, vec2, axis=-1, epsilon=1e-8):
    """Compute cosine similarity between two velocity fields."""
    vec1 = vec1.to(dtype=torch.float32, device=vec2.device)
    vec2 = vec2.to(dtype=torch.float32, device=vec2.device)

    if vec1.shape != vec2.shape:
        raise ValueError(f"Shape mismatch: vec1 {vec1.shape}, vec2 {vec2.shape}")

    if vec1.dim() > 2:
        vec1_flat = vec1.view(vec1.shape[0], -1)
        vec2_flat = vec2.view(vec2.shape[0], -1)
        sim = F.cosine_similarity(vec1_flat, vec2_flat, dim=axis, eps=epsilon)
    else:
        sim = F.cosine_similarity(vec1, vec2, dim=axis, eps=epsilon)

    return sim

def compute_shift_factor(seq_length, base_len=256, max_len=4096, min_shift=0.5, max_shift=1.16):
    """Compute shift factor for diffusion process."""
    slope = (max_shift - min_shift) / (max_len - base_len)
    intercept = min_shift - slope * base_len
    return seq_length * slope + intercept

def calculate_velocity(pipeline, latents, embeddings, pooled_embeds, guidance, text_ids, img_ids, step_time, img_height):
    """Calculate velocity field for transformer model."""
    step_time = step_time.expand(latents.shape[0])
    with torch.no_grad():
        prediction = pipeline.transformer(
            hidden_states=latents,
            timestep=step_time / 1000,
            guidance=guidance,
            encoder_hidden_states=embeddings,
            txt_ids=text_ids,
            img_ids=img_ids,
            pooled_projections=pooled_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
            height=img_height // 16
        )[0]
    return prediction


def CausalEdit(
        pipeline,
        scheduler,
        input_latent,
        src_prompt,
        tar_prompt,
        num_steps,
        src_guidance,
        tar_guidance,
        Lambda,
        num_samples: int = 0
):
    """
    Edit input latent using transformer pipeline and captions.

    Args:
        pipeline:
        scheduler: Scheduler for diffusion process control.
        input_latent: Input latent representation.
        src_prompt: Original image description.
        tar_prompt: Target description.
        num_steps: Number of edit steps
        src_guidance: Guidance scale for source prompt
        tar_guidance: Guidance scale for target prompt
        Lambda: Weight for Lambda parameter
        num_samples: Number of additional sampling points

    Returns:
        latent representation.
    """
    device = input_latent.device
    img_height, img_width = input_latent.shape[2] * pipeline.vae_scale_factor // 2, input_latent.shape[3] * pipeline.vae_scale_factor // 2
    channel_count = pipeline.transformer.config.in_channels // 4

    pipeline.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=img_height,
        width=img_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512
    )

    # Prepare latents
    latents, orig_img_ids = pipeline.prepare_latents(
        batch_size=input_latent.shape[0],
        num_channels_latents=channel_count,
        height=img_height,
        width=img_width,
        dtype=input_latent.dtype,
        device=device,
        generator=None,
        latents=input_latent
    )
    packed_latents = pipeline._pack_latents(latents, latents.shape[0], channel_count, latents.shape[2], latents.shape[3])
    new_img_ids = orig_img_ids

    total_steps = num_steps + 5
    sigma_values = np.linspace(1.0, 1 / total_steps, total_steps)
    shift_val = compute_shift_factor(
        packed_latents.shape[1],
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift
    )
    timesteps, _ = retrieve_timesteps(scheduler, total_steps, device, timesteps=None, sigmas=sigma_values, mu=shift_val)

    timesteps = timesteps[5:] if len(timesteps) > num_steps else timesteps
    pipeline._num_timesteps = len(timesteps)

    orig_embeds, orig_pooled, orig_text_ids = pipeline.encode_prompt(
        prompt=src_prompt, prompt_2=None, device=device
    )
    pipeline._guidance_scale = tar_guidance
    tar_embeds, tar_pooled, new_text_ids = pipeline.encode_prompt(
        prompt=tar_prompt, prompt_2=None, device=device
    )

    # Set guidance tensors
    orig_guidance_tensor = torch.tensor([src_guidance], device=device).expand(packed_latents.shape[0]) if pipeline.transformer.config.guidance_embeds else None
    tar_guidance_tensor = torch.tensor([tar_guidance], device=device).expand(packed_latents.shape[0]) if pipeline.transformer.config.guidance_embeds else None

    # Line 1: Initialize target latent (x_t = x_max)
    updated_latent = packed_latents.clone()

    torch.cuda.empty_cache()

    # Line 2: Loop over timesteps (while t >= t_0)
    for idx, step_time in tqdm(enumerate(timesteps), total=len(timesteps)):
        scheduler._init_step_index(step_time)
        curr_sigma = scheduler.sigmas[scheduler.step_index]
        next_sigma = scheduler.sigmas[scheduler.step_index + 1] if idx < len(timesteps) - 1 else curr_sigma

        velocity_sum = torch.zeros_like(packed_latents)
        noise = torch.randn_like(packed_latents).to(device)

        # Line 3: Apply Gaussian noise injection (z_t = (1 - t) * x_t + t * ε)
        orig_zt = (1 - curr_sigma) * packed_latents + curr_sigma * noise

        # Target latent adjustment
        tar_zt = updated_latent + orig_zt - packed_latents

        # Line 4: Compute base velocities (v_t = Φ(z_t, P_tar, t))
        tar_velocity = calculate_velocity(
            pipeline, tar_zt, tar_embeds, tar_pooled,
            tar_guidance_tensor, new_text_ids, new_img_ids, step_time, img_height
        )
        # Lines 5-10: Sample and compute additional velocities (x_t^k = x_src + (jk-1)*v_t + ε, P_t^k = jk*P_tar + (1-jk)*P_src, v_t^k = Φ(x_t^k, P_t^k, t))
        ### additional samplings
        sample_velocities = []
        for i in range(1, num_samples + 1):
            weight = i / (num_samples + 2)
            sample_xt = packed_latents + noise - weight * tar_velocity
            sample_embeds = weight * tar_embeds + (1 - weight) * orig_embeds
            sample_pooled = weight * tar_pooled + (1 - weight) * orig_pooled
            sample_velocity = calculate_velocity(
                pipeline, sample_xt, sample_embeds, sample_pooled,
                tar_guidance_tensor, new_text_ids, new_img_ids, step_time, img_height
            )
            sample_velocities.append(sample_velocity)
        orig_velocity = calculate_velocity(
            pipeline, orig_zt, orig_embeds, orig_pooled,
            orig_guidance_tensor, orig_text_ids, orig_img_ids, step_time, img_height
        )
        ### additional samplings

        # Line 11: Aggregate velocities (v_t = v_t + λ * Σ I(k) * v_t^k)
        velocity_sum += tar_velocity
        for i, velocity in enumerate(sample_velocities):
            coefficient = Lambda if i % 2 == 0 else -Lambda
            velocity_sum += coefficient * velocity
        velocity_sum += - Lambda * orig_velocity

        # Line 12: Update latent (x_{t-1} = z_t + (t_{t-1} - t) * v_t)
        updated_latent += (next_sigma - curr_sigma) * velocity_sum

        # Line 13: Update timestep (t = t-1, handled by loop)

    # Line 14: Return final latent (return x_t)
    final_output = pipeline._unpack_latents(updated_latent, img_height, img_width, pipeline.vae_scale_factor)
    torch.cuda.empty_cache()
    return final_output

def configure_logging(save_dir):
    """Set up logging to file and console."""
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{save_dir}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def setup_model(model_dir, device):
    """Initialize model pipeline."""
    pipeline = FluxPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    pipeline = initialize_pipeline(pipeline)
    pipeline.enable_sequential_cpu_offload()
    return pipeline

def define_image_transform(target_height, target_width):
    """Create image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def edit_image(model_pipeline, img_path, orig_caption, new_caption, params, save_dir, device):
    """Process a single image with causal editing."""
    torch.cuda.empty_cache()
    start_time = time()

    # Load and preprocess image
    img = Image.open(img_path)
    transform_fn = define_image_transform(params.height, params.width)
    img_tensor = transform_fn(img).unsqueeze(0).to(device)

    # Preprocess image and encode to latent space
    processed_img = model_pipeline.image_processor.preprocess(img_tensor)
    processed_img = processed_img.to(device).half()

    with torch.autocast("cuda"), torch.inference_mode():
        latent_rep = model_pipeline.vae.encode(processed_img).latent_dist.mode().to(device)

    adjusted_latent = (latent_rep - model_pipeline.vae.config.shift_factor) * model_pipeline.vae.config.scaling_factor
    adjusted_latent = adjusted_latent.to(device)

    # Generate task ID from image filename
    task_id = os.path.basename(img_path).split('.')[0]

    # Apply causal editing
    update_latent = CausalEdit(
        model_pipeline,
        model_pipeline.scheduler,
        adjusted_latent,
        orig_caption,
        new_caption,
        num_steps=25,
        src_guidance=params.src_guidance,
        tar_guidance=params.tar_guidance,
        Lambda=params.Lambda,
        num_samples=params.num_samples
    )

    # Decode latent back to image
    restored_latent = (update_latent / model_pipeline.vae.config.scaling_factor) + model_pipeline.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        final_image = model_pipeline.vae.decode(restored_latent, return_dict=False)[0]
    final_image = model_pipeline.image_processor.postprocess(final_image)[0]

    # Save output image
    os.makedirs(save_dir, exist_ok=True)
    output_path = f"{save_dir}/{task_id}.png"
    final_image.save(output_path)

    duration = time() - start_time
    logging.info(f"Image processed: {task_id}, saved to {output_path}, Duration: {duration:.0f} seconds")

    torch.cuda.empty_cache()
    return final_image

def run_workflow(params):
    """Main execution workflow for single image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_logging(params.save_dir)

    # Initialize model
    model_pipeline = setup_model(params.model_dir, device)

    # Process the single image
    try:
        edited_img = edit_image(
            model_pipeline,
            params.input_image,
            params.source_prompt,
            params.target_prompt,
            params,
            params.save_dir,
            device
        )
    except Exception as e:
        logging.error(f"Image processing failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single image editing using Flux model')
    parser.add_argument('--model_dir', type=str, default='FLUX.1-dev', help='Path to model weights')
    parser.add_argument('--input_image', type=str, default='horse.jpg', help='Path to input image')
    parser.add_argument('--source_prompt', type=str, default="A horse", help='Original image description')
    parser.add_argument('--target_prompt', type=str, default="A zebra", help='Target description')
    parser.add_argument('--height', type=int, default=1024, help='Output image height')
    parser.add_argument('--width', type=int, default=1024, help='Output image width')
    parser.add_argument('--Lambda', type=float, default=1, help='Lambda parameter')
    parser.add_argument('--src_guidance', type=float, default=1.5, help='source prompt guidance scale')
    parser.add_argument('--tar_guidance', type=float, default=5.5, help='target prompt guidance scale')
    parser.add_argument('--save_dir', type=str, default='outputs/outputs', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=0, help='Number of additional sampling points')
    params = parser.parse_args()
    run_workflow(params)
