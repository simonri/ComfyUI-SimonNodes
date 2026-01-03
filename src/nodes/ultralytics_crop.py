import torch

from ultralytics import YOLO

from ..main import crop_image_to_pose
from comfy.utils import common_upscale

TARGET_WIDTH = 768
TARGET_HEIGHT = 848

class UltralyticsCrop:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "image": ("IMAGE",),
        "ultralytics_model": ("ULTRALYTICS_MODEL",),
        "target_width": ("INT", {"default": TARGET_WIDTH, "min": 0, "max": 10000, "step": 1}),
        "target_height": ("INT", {"default": TARGET_HEIGHT, "min": 0, "max": 10000, "step": 1}),
      },
    }

  RETURN_TYPES = ("IMAGE", "IMAGE")
  RETURN_NAMES = ("cropped_image", "upscaled_image")
  FUNCTION = "process"
  CATEGORY = "Simon"

  def process(self, image: torch.Tensor, ultralytics_model: YOLO, target_width: int, target_height: int):
    # image: [B, H, W, C]
    image = image.movedim(-1, 1) # [B, C, H, W]

    cropped_image = crop_image_to_pose(image, ultralytics_model, target_width, target_height)
    upscaled_image = common_upscale(cropped_image, target_width, target_height, "lanczos", "center")

    cropped_image = cropped_image.movedim(1, -1) # [B, H, W, C]
    upscaled_image = upscaled_image.movedim(1, -1) # [B, H, W, C]

    return (cropped_image, upscaled_image)


NODE_CLASS_MAPPINGS = {
  "UltralyticsCrop": UltralyticsCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "UltralyticsCrop": "Ultralytics Crop",
}
