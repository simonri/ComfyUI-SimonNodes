from comfy.ldm.wan.vae import Encoder3d, Decoder3d, CausalConv3d, count_conv3d
import torch
import torch.nn as nn


class WanVAE(nn.Module):
  def __init__(
    self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], in_channels=3, out_channels=3, dropout=0.0
  ):
    super().__init__()
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_downsample = temperal_downsample
    self.temperal_upsample = temperal_downsample[::-1]

    # modules
    self.encoder = Encoder3d(dim, z_dim * 2, in_channels, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout)
    self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
    self.conv2 = CausalConv3d(z_dim, z_dim, 1)
    self.decoder = Decoder3d(dim, z_dim, out_channels, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

  def encode(self, x):
    conv_idx = [0]
    ## cache
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    feat_map = None
    if iter_ > 1:
      feat_map = [None] * count_conv3d(self.decoder)
    ## 对encode输入的x，按时间拆分为1、4、4、4....
    for i in range(iter_):
      conv_idx = [0]
      if i == 0:
        out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
      else:
        out_ = self.encoder(x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=feat_map, feat_idx=conv_idx)
        out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    return mu

  def decode(self, z):
    conv_idx = [0]
    # z: [b,c,t,h,w]
    iter_ = z.shape[2]
    feat_map = None
    if iter_ > 1:
      feat_map = [None] * count_conv3d(self.decoder)
    x = self.conv2(z)
    for i in range(iter_):
      conv_idx = [0]
      if i == 0:
        out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
      else:
        out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
        out = torch.cat([out, out_], 2)
    return out
