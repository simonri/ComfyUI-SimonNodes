import math
import torch
import logging

from comfy import model_management
from comfy.ldm.models.autoencoder import AutoencoderKL, AutoencodingEngine

from .wan_vae import WanVAE
import comfy.ldm.wan.vae2_2
import comfy.utils

import comfy.model_patcher
import comfy.taesd.taesd

from comfy.sd import VAE


class CustomVAE(VAE):
  def __init__(self, sd=None, device=None, config=None, dtype=None, metadata=None):
    if model_management.is_amd():
      VAE_KL_MEM_RATIO = 2.73
    else:
      VAE_KL_MEM_RATIO = 1.0

    self.memory_used_encode = (
      lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO
    )  # These are for AutoencoderKL and need tweaking (should be lower)
    self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype) * VAE_KL_MEM_RATIO
    self.downscale_ratio = 8
    self.upscale_ratio = 8
    self.latent_channels = 4
    self.latent_dim = 2
    self.input_channels = 3
    self.output_channels = 3
    self.real_output_channels = 3
    self.pad_channel_value = None
    self.process_input = lambda image: image * 2.0 - 1.0
    self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    self.working_dtypes = [torch.bfloat16, torch.float32]
    self.disable_offload = False
    self.not_video = False
    self.size = None

    self.downscale_index_formula = None
    self.upscale_index_formula = None
    self.extra_1d_channel = None
    self.crop_input = True

    if config is None:
      if "decoder.mid.block_1.mix_factor" in sd:
        encoder_config = {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        }
        decoder_config = encoder_config.copy()
        decoder_config["video_kernel_size"] = [3, 1, 1]
        decoder_config["alpha"] = 0.0
        self.first_stage_model = AutoencodingEngine(
          regularizer_config={"target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
          encoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Encoder", "params": encoder_config},
          decoder_config={"target": "comfy.ldm.modules.temporal_ae.VideoDecoder", "params": decoder_config},
        )
      elif "taesd_decoder.1.weight" in sd:
        self.latent_channels = sd["taesd_decoder.1.weight"].shape[1]
        self.first_stage_model = comfy.taesd.taesd.TAESD(latent_channels=self.latent_channels)

      elif "decoder.conv_in.weight" in sd:
        ddconfig = {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        }

        if "encoder.down.2.downsample.conv.weight" not in sd and "decoder.up.3.upsample.conv.weight" not in sd:  # Stable diffusion x4 upscaler VAE
          ddconfig["ch_mult"] = [1, 2, 4]
          self.downscale_ratio = 4
          self.upscale_ratio = 4

        self.latent_channels = ddconfig["z_channels"] = sd["decoder.conv_in.weight"].shape[1]
        if "post_quant_conv.weight" in sd:
          self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1])
        else:
          self.first_stage_model = AutoencodingEngine(
            regularizer_config={"target": "comfy.ldm.models.autoencoder.DiagonalGaussianRegularizer"},
            encoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Encoder", "params": ddconfig},
            decoder_config={"target": "comfy.ldm.modules.diffusionmodules.model.Decoder", "params": ddconfig},
          )
      elif "decoder.conv_in.conv.weight" in sd:
        ddconfig = {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1, 2, 4, 4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0,
        }
        ddconfig["conv3d"] = True
        ddconfig["time_compress"] = 4
        self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
        self.upscale_index_formula = (4, 8, 8)
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
        self.downscale_index_formula = (4, 8, 8)
        self.latent_dim = 3
        self.latent_channels = ddconfig["z_channels"] = sd["decoder.conv_in.conv.weight"].shape[1]
        self.first_stage_model = AutoencoderKL(ddconfig=ddconfig, embed_dim=sd["post_quant_conv.weight"].shape[1])
        self.memory_used_decode = lambda shape, dtype: (1500 * shape[2] * shape[3] * shape[4] * (4 * 8 * 8)) * model_management.dtype_size(dtype)
        self.memory_used_encode = lambda shape, dtype: (900 * max(shape[2], 2) * shape[3] * shape[4]) * model_management.dtype_size(dtype)
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
      elif "decoder.middle.0.residual.0.gamma" in sd:
        if "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd:  # Wan 2.2 VAE
          self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 16, 16)
          self.upscale_index_formula = (4, 16, 16)
          self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 16, 16)
          self.downscale_index_formula = (4, 16, 16)
          self.latent_dim = 3
          self.latent_channels = 48
          ddconfig = {
            "dim": 160,
            "z_dim": self.latent_channels,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
          }
          self.first_stage_model = comfy.ldm.wan.vae2_2.WanVAE(**ddconfig)
          self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
          self.memory_used_encode = lambda shape, dtype: 3300 * shape[3] * shape[4] * model_management.dtype_size(dtype)
          self.memory_used_decode = lambda shape, dtype: 8000 * shape[3] * shape[4] * (16 * 16) * model_management.dtype_size(dtype)
        else:  # Wan 2.1 VAE
          dim = sd["decoder.head.0.gamma"].shape[0]
          self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
          self.upscale_index_formula = (4, 8, 8)
          self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
          self.downscale_index_formula = (4, 8, 8)
          self.input_channels = sd["encoder.conv1.weight"].shape[1]
          # self.output_channels = sd["decoder.head.2.weight"].shape[0]
          self.output_channels = self.input_channels
          self.real_output_channels = sd["decoder.head.2.weight"].shape[0]
          self.latent_dim = 3
          self.latent_channels = 16
          self.pad_channel_value = 1.0

          ddconfig = {
            "in_channels": self.input_channels,
            "out_channels": self.real_output_channels,
            "dim": dim,
            "z_dim": self.latent_channels,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
          }
          self.first_stage_model = WanVAE(**ddconfig)
          self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
          self.memory_used_encode = lambda shape, dtype: 6000 * shape[3] * shape[4] * model_management.dtype_size(dtype)
          self.memory_used_decode = lambda shape, dtype: 7000 * shape[3] * shape[4] * (8 * 8) * model_management.dtype_size(dtype)
      else:
        logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
        self.first_stage_model = None
        return
    else:
      self.first_stage_model = AutoencoderKL(**(config["params"]))
    self.first_stage_model = self.first_stage_model.eval()

    m, u = self.first_stage_model.load_state_dict(sd, strict=False)
    if len(m) > 0:
      logging.warning("Missing VAE keys {}".format(m))

    if len(u) > 0:
      logging.debug("Leftover VAE keys {}".format(u))

    if device is None:
      device = model_management.vae_device()
    self.device = device
    offload_device = model_management.vae_offload_device()
    if dtype is None:
      dtype = model_management.vae_dtype(self.device, self.working_dtypes)
    self.vae_dtype = dtype
    self.first_stage_model.to(self.vae_dtype)
    self.output_device = model_management.intermediate_device()

    self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
    logging.info("VAE load device: {}, offload device: {}, dtype: {}".format(self.device, offload_device, self.vae_dtype))
    self.model_size()
