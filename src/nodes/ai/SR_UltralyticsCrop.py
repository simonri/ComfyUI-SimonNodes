import torch
from ultralytics import YOLO
import torch.nn.functional as F

from comfy.utils import common_upscale

TARGET_WIDTH = 768
TARGET_HEIGHT = 848

HIP_CONF_THRESHOLD = 0.3
YOLO_INPUT_W = 640
YOLO_INPUT_H = 640


def crop_image_to_pose(image: torch.Tensor, ultralytics_model: YOLO, target_w, target_h):
  # image: [B, C, H, W], range [0,1]
  assert image.ndim == 4 and image.shape[0] == 1

  print(f"target_w: {target_w}, target_h: {target_h}")

  _, C, orig_h, orig_w = image.shape

  img_chw = image[0]
  img_resized = F.interpolate(img_chw.unsqueeze(0), size=(YOLO_INPUT_H, YOLO_INPUT_W), mode="bilinear", align_corners=False)

  results = ultralytics_model(img_resized)

  result = results[0]

  scale_x = orig_w / YOLO_INPUT_W
  scale_y = orig_h / YOLO_INPUT_H

  if len(result.boxes) == 0:
    return image

  boxes = result.boxes
  keypoints = result.keypoints.data

  idx = boxes.conf.argmax().item()

  x1, y1, x2, y2 = boxes.xyxy[idx]
  person_kpts = keypoints[idx]

  # Scale
  person_kpts = person_kpts.clone()
  person_kpts[:, 0] *= scale_x
  person_kpts[:, 1] *= scale_y
  box_xyxy = boxes.xyxy[idx].clone()
  box_xyxy *= torch.tensor([scale_x, scale_y, scale_x, scale_y], device=box_xyxy.device)
  x1, y1, x2, y2 = box_xyxy

  # Get nose y
  nose_y = person_kpts[0][1]

  upper_bound_y = max(0, int(y1 - (nose_y - y1) * 0.5))
  lower_bound_y = orig_h

  has_hip = person_kpts[11][2] > HIP_CONF_THRESHOLD and person_kpts[12][2] > HIP_CONF_THRESHOLD
  if has_hip:
    hip_y = (person_kpts[11][1] + person_kpts[12][1]) * 0.5
    print(f"hip_y: {hip_y}")
    lower_bound_y = int(min(lower_bound_y, hip_y - (hip_y - nose_y) * 0.4))
  print(f"lower_bound_y: {lower_bound_y}")

  # Get shoulder center x
  x_center = (person_kpts[5][0] + person_kpts[6][0]) * 0.5

  # Start with crop_h = 0 and increase
  crop_h = 0
  target_aspect = target_w / target_h

  def is_left_bound_hit(x_center, crop_w):
    return x_center - crop_w * 0.5 < 0

  def is_right_bound_hit(x_center, crop_w):
    return x_center + crop_w * 0.5 > orig_w

  def is_upper_bound_hit(y0_crop):
    return y0_crop < upper_bound_y

  def is_lower_bound_hit(y1_crop):
    return y1_crop > lower_bound_y

  max_crop_h = orig_h

  while crop_h < max_crop_h:
    crop_h += 2
    crop_w = crop_h * target_aspect

    y0_crop = nose_y - crop_h * 0.5
    y1_crop = nose_y + crop_h * 0.5

    left_bound_hit = is_left_bound_hit(x_center, crop_w)
    right_bound_hit = is_right_bound_hit(x_center, crop_w)
    upper_bound_hit = is_upper_bound_hit(y0_crop)
    lower_bound_hit = is_lower_bound_hit(y1_crop)

    # Check all three conditions
    if lower_bound_hit:
      crop_h -= 2
      break
    if left_bound_hit and right_bound_hit:
      crop_h -= 2
      break

    if upper_bound_hit and not is_lower_bound_hit(nose_y + 1 + crop_h * 0.5):
      nose_y += 1

    if left_bound_hit and not is_right_bound_hit(x_center + 1, crop_w):
      x_center += 1
    if right_bound_hit and not is_left_bound_hit(x_center - 1, crop_w):
      x_center -= 1

  crop_w = int(crop_h * target_aspect)
  y0_crop = nose_y - crop_h * 0.5

  print(f"y0_crop: {y0_crop}, crop_h: {crop_h}, crop_w: {crop_w}")

  print(f"crop: {crop_w / crop_h}, target_aspect: {target_aspect}")

  left = int(x_center - crop_w * 0.5)
  top = int(y0_crop)

  print(f"left: {left}, top: {top}, crop_w: {crop_w}, crop_h: {crop_h}")

  cropped = image[:, :, top : top + crop_h, left : left + crop_w]

  # return [B, C, H, W]
  return cropped


class SR_UltralyticsCrop:
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
    image = image.movedim(-1, 1)  # [B, C, H, W]

    cropped_image = crop_image_to_pose(image, ultralytics_model, target_width, target_height)
    upscaled_image = common_upscale(cropped_image, target_width, target_height, "lanczos", "center")

    cropped_image = cropped_image.movedim(1, -1)  # [B, H, W, C]
    upscaled_image = upscaled_image.movedim(1, -1)  # [B, H, W, C]

    return (cropped_image, upscaled_image)


NODE_CLASS_MAPPINGS = {
  "SR_UltralyticsCrop": SR_UltralyticsCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "SR_UltralyticsCrop": "SR Ultralytics Crop",
}
