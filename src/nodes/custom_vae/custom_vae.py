import torch
import torch.nn.functional as F
import copy
import math

import comfy.utils
import comfy.model_management
import comfy.latent_formats
import folder_paths
from nodes import VAELoader
from .sd import CustomVAE
from .model import latent_upscale_models


class VAEUtils_CustomVAELoader(VAELoader):
  @staticmethod
  def vae_list():
    return folder_paths.get_filename_list("vae")

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "vae_name": (s.vae_list(),),
        "disable_offload": ("BOOLEAN", {"default": True}),
      }
    }

  RETURN_TYPES = ("VAE",)
  FUNCTION = "load_vae"
  CATEGORY = "Simon"

  def load_vae(self, vae_name, disable_offload):
    if vae_name == "pixel_space":
      sd = {}
      sd["pixel_space_vae"] = torch.tensor(1.0)
    elif vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
      sd = self.load_taesd(vae_name)
    else:
      vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
      sd = comfy.utils.load_torch_file(vae_path)

    vae = CustomVAE(sd=sd)
    vae.throw_exception_if_invalid()
    vae.disable_offload = disable_offload
    return (vae,)


class VAEUtils_DisableVAEOffload:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "vae": ("VAE",),
        "disable_offload": ("BOOLEAN", {"default": True}),
      }
    }

  RETURN_TYPES = ("VAE",)
  FUNCTION = "set_offload"
  CATEGORY = "Simon"

  def set_offload(self, vae, disable_offload):
    vae = copy.copy(vae)
    vae.disable_offload = disable_offload
    return (vae,)


class VAEUtils_VAEDecodeTiled:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "upscale": ("INT", {"default": -1, "min": -1, "tooltip": "Post upscale factor, -1=auto"})}}

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "decode"
  CATEGORY = "Simon"

  def decode(self, samples, vae, upscale):
    latent = samples["samples"]
    if latent.is_nested:
      latent = latent.unbind()[0]

    images = vae.decode(latent)

    if len(images.shape) == 5:  # Combine batches
      images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    if upscale < 1:
      ch = images.shape[-1]
      if ch == 3:
        upscale = 1
      else:
        if ch % 3 == 0:
          upscale = round((ch // 3) ** 0.5)
        else:
          raise Exception("Couldn't determine upscale factor, try setting the value manually instead")

    images = F.pixel_shuffle(images.movedim(-1, 1), upscale_factor=int(upscale)).movedim(1, -1)
    return (images,)


class VAEUtils_LatentUpscale:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "samples": ("LATENT",),
        "model": (list(latent_upscale_models.keys()),),
      }
    }

  RETURN_TYPES = ("LATENT",)
  FUNCTION = "upscale"
  CATEGORY = "Simon"

  def upscale(self, samples, model):
    device = comfy.model_management.get_torch_device()
    model = latent_upscale_models[model]().to(device)

    latents = samples["samples"].to(dtype=torch.float32, device=device)
    upscaled_latents = model(latents).to(comfy.model_management.intermediate_device())

    samples = copy.deepcopy(samples)
    samples["samples"] = upscaled_latents

    return (samples,)


def get_tiles(length, tile_size, min_overlap):
  if length <= tile_size:
    return [(0, length)]

  max_step = tile_size - min_overlap
  total_shiftable = length - tile_size

  gaps_needed = math.ceil(total_shiftable / max_step) if total_shiftable > 0 else 0
  num_tiles = gaps_needed + 1

  if num_tiles == 1:
    raise Exception("this shouldn't happen")

  gap_base = total_shiftable // (num_tiles - 1)
  remainder = total_shiftable % (num_tiles - 1)

  starts = []
  acc = 0
  for i in range(num_tiles):
    starts.append(acc)
    if i < num_tiles - 1:
      acc += gap_base + (1 if i < remainder else 0)

  slices = [(s, s + tile_size) for s in starts]
  return slices


def get_1d_mask(idx, tiles, drop_first=0):
  tile_start, tile_end = tiles[idx]
  mask = torch.ones(tile_end - tile_start)

  if idx > 0:
    prev_end = tiles[idx - 1][1]
    size = prev_end - tile_start - drop_first
    ramp = (torch.arange(size) + 1) / size
    mask[drop_first : size + drop_first] *= ramp
    mask[:drop_first] *= 0

  if idx < (len(tiles) - 1):
    next_start = tiles[idx + 1][0]
    size = tile_end - next_start
    ramp = (torch.flip(torch.arange(size), [0]) + 1) / size
    mask[-size:] *= ramp

  return mask


latent_formats = {name: obj for name, obj in vars(comfy.latent_formats).items() if isinstance(obj, type)}
del latent_formats["LatentFormat"]


class VAEUtils_ScaleLatents:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "latents": ("LATENT",),
        "direction": (["scale", "unscale"],),
        "latent_type": (list(latent_formats.keys()),),
      }
    }

  RETURN_TYPES = ("LATENT",)
  FUNCTION = "scale"
  CATEGORY = "Simon"

  def scale(self, latents, direction, latent_type):
    latents = copy.deepcopy(latents)
    latent_format = latent_formats[latent_type]()

    if direction == "scale":
      latents["samples"] = latent_format.process_in(latents["samples"])
    else:
      latents["samples"] = latent_format.process_out(latents["samples"])

    return (latents,)


NODE_CLASS_MAPPINGS = {
  "VAEUtils_CustomVAELoader": VAEUtils_CustomVAELoader,
  "VAEUtils_DisableVAEOffload": VAEUtils_DisableVAEOffload,
  "VAEUtils_VAEDecodeTiled": VAEUtils_VAEDecodeTiled,
  "VAEUtils_LatentUpscale": VAEUtils_LatentUpscale,
  "VAEUtils_ScaleLatents": VAEUtils_ScaleLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "VAEUtils_CustomVAELoader": "Load VAE (VAE Utils)",
  "VAEUtils_DisableVAEOffload": "Disable VAE Offload (VAE Utils)",
  "VAEUtils_VAEDecodeTiled": "VAE Decode (VAE Utils)",
  "VAEUtils_LatentUpscale": "Latent Upscale (VAE Utils)",
  "VAEUtils_ScaleLatents": "Scale/Unscale Latents (VAE Utils)",
}
