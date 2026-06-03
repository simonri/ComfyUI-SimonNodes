import os
import hashlib
import logging

import torch

import folder_paths

# Bump this if the on-disk format ever changes, to invalidate old caches.
CACHE_VERSION = "v1"
CACHE_DIR = os.path.join(folder_paths.base_path, "cache", "simon_text_encode")


def _cache_path(text: str, cache_id: str) -> str:
  # Key is computed WITHOUT the CLIP model so it can be evaluated before the
  # model is (lazily) loaded. cache_id lets the user namespace per CLIP model
  # so swapping models with identical text doesn't reuse stale embeddings.
  raw = "\x00".join([CACHE_VERSION, cache_id, text]).encode("utf-8")
  key = hashlib.sha256(raw).hexdigest()
  return os.path.join(CACHE_DIR, f"{key}.pt")


def _to_cpu(conditioning):
  # conditioning is list[[cond_tensor, dict]]; move every tensor to CPU so the
  # cache is portable and doesn't pin VRAM. Sampling moves conds to the device.
  out = []
  for cond, meta in conditioning:
    cond_cpu = cond.detach().cpu() if torch.is_tensor(cond) else cond
    meta_cpu = {
      k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in meta.items()
    }
    out.append([cond_cpu, meta_cpu])
  return out


def _load_cache(path: str):
  try:
    if os.path.exists(path):
      return torch.load(path, map_location="cpu", weights_only=False)
  except Exception as e:
    logging.warning(f"[SR_CLIPTextEncodeCached] Failed to load cache {path}: {e}")
  return None


def _save_cache(path: str, conditioning) -> None:
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_to_cpu(conditioning), path)
    logging.info(f"[SR_CLIPTextEncodeCached] Saved encode cache: {path}")
  except Exception as e:
    logging.warning(f"[SR_CLIPTextEncodeCached] Failed to save cache {path}: {e}")


class SR_CLIPTextEncodeCached:
  """Like CLIPTextEncode, but caches the encoded conditioning to disk.

  On a cache hit the CLIP text encoder is never loaded: the `clip` input is
  declared lazy and check_lazy_status returns [] so ComfyUI does not schedule
  the upstream CLIP loader.
  """

  NAME = "SR CLIP Text Encode (Cached)"
  CATEGORY = "Simon"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
        "clip": ("CLIP", {"lazy": True, "tooltip": "The CLIP model used for encoding the text. Not loaded when a cached encode exists."}),
      },
      "optional": {
        "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Read/write the on-disk encode cache. When off, behaves exactly like CLIPTextEncode."}),
        "cache_id": ("STRING", {"default": "", "tooltip": "Namespace for the cache, e.g. the CLIP model name. Change it when you change CLIP models so stale embeddings aren't reused."}),
      },
    }

  RETURN_TYPES = ("CONDITIONING",)
  RETURN_NAMES = ("CONDITIONING",)
  OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
  FUNCTION = "encode"
  DESCRIPTION = "Encodes a text prompt with CLIP into conditioning, caching the result to disk. On a cache hit the CLIP text encoder is not loaded."

  def __init__(self):
    self._cached = None
    self._cached_path = None

  def check_lazy_status(self, text, clip=None, use_cache=True, cache_id=""):
    # Decide whether the (lazy) clip input needs to be resolved. Returning []
    # means "don't need it", which prevents the upstream CLIP loader from
    # running at all. Always recompute here so stale state can't leak between
    # runs of a reused node instance.
    self._cached = None
    self._cached_path = _cache_path(text, cache_id)

    if use_cache:
      self._cached = _load_cache(self._cached_path)
      if self._cached is not None:
        return []  # cache hit -> do not load the model

    # Cache miss (or caching off): we need clip, unless it's already resolved.
    return ["clip"] if clip is None else []

  def encode(self, text, clip=None, use_cache=True, cache_id=""):
    # Always derive the path from this call's own inputs; never trust state
    # left over from a check_lazy_status call for different text.
    path = _cache_path(text, cache_id)

    if use_cache:
      # Reuse the stash only if it was loaded for exactly this path.
      cached = self._cached if self._cached_path == path else None
      if cached is None:
        cached = _load_cache(path)
      if cached is not None:
        return (cached,)

    if clip is None:
      raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    tokens = clip.tokenize(text)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if use_cache:
      _save_cache(path, conditioning)

    return (conditioning,)


NODE_CLASS_MAPPINGS = {"SR_CLIPTextEncodeCached": SR_CLIPTextEncodeCached}
NODE_DISPLAY_NAME_MAPPINGS = {"SR_CLIPTextEncodeCached": SR_CLIPTextEncodeCached.NAME}
