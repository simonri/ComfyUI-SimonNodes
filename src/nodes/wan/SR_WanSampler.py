# class WanSampler:
#   @classmethod
#   def INPUT_TYPES(s):
#     return {
#       "model": ("MODEL",),
#       "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
#       "steps": ("INT", {"default": 30, "min": 1}),
#       "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
#       "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
#       "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#       "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
#       "scheduler": (scheduler_list, {"default": "unipc",})
#     }

#   RETURN_TYPES = ("LATENT", "LATENT")
#   RETURN_NAMES = ("samples", "denoised_samples")
#   FUNCTION = "process"
#   CATEGORY = "Simon"

#   def process(self, model, image_embeds, steps, cfg, shift, seed, force_offload, scheduler):

import torch
import comfy.sample
import comfy.utils
import latent_preview

scheduler_list = ["euler"]

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
  latent_image = latent["samples"]
  latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))

  if disable_noise:
    noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
  else:
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

  noise_mask = None
  if "noise_mask" in latent:
    noise_mask = latent["noise_mask"]

  callback = latent_preview.prepare_callback(model, steps)
  disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
  samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
  out = latent.copy()
  out.pop("downscale_ratio_spacial", None)
  out["samples"] = samples
  return (out, )


class SR_WanSampler:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "model": ("MODEL",),
        "add_noise": (["enable", "disable"], ),
        "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
        "scheduler": (scheduler_list, {"default": "euler"}),
        "positive": ("CONDITIONING", ),
        "negative": ("CONDITIONING", ),
        "latent_image": ("LATENT", ),
        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
        "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
        "return_with_leftover_noise": (["disable", "enable"], ),
      }
    }

  RETURN_TYPES = ("LATENT",)
  FUNCTION = "sample"
  CATEGORY = "Simon"

  def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
    force_full_denoise = True
    if return_with_leftover_noise == "enable":
      force_full_denoise = False
    disable_noise = False
    if add_noise == "disable":
      disable_noise = True
    return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


NODE_CLASS_MAPPINGS = {
  "SR_WanSampler": SR_WanSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "SR_WanSampler": "SR Wan Sampler",
}
