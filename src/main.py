from ultralytics import YOLO
import torch
import torch.nn.functional as F

HIP_CONF_THRESHOLD = 0.3
YOLO_INPUT_W = 640
YOLO_INPUT_H = 640

def fix_scale_x(x, scale_x):
  return int(x * scale_x)

def fix_scale_y(y, scale_y):
  return int(y * scale_y)


def crop_image_to_pose(image: torch.Tensor, ultralytics_model: YOLO, target_w, target_h):
  # image: [B, C, H, W], range [0,1]
  assert image.ndim == 4 and image.shape[0] == 1

  print(f"target_w: {target_w}, target_h: {target_h}")

  _, C, orig_h, orig_w = image.shape

  img_chw = image[0]
  img_resized = F.interpolate(
    img_chw.unsqueeze(0),
    size=(YOLO_INPUT_W, YOLO_INPUT_H),
    mode="bilinear",
    align_corners=False
  )

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

  # Get nose y
  nose_y = fix_scale_y(person_kpts[0][1], scale_y)
  # nose_x = fix_scale_x(person_kpts[0][0], scale_x)

  # Set nose coord to red
  # image[0, :, nose_y, nose_x] = torch.tensor([1, 0, 0])
  # image[0, :, nose_y-2, nose_x] = torch.tensor([1, 0, 0])
  # image[0, :, nose_y-4, nose_x] = torch.tensor([1, 0, 0])

  # Get shoulder center x
  x_center = fix_scale_x((person_kpts[5][0] + person_kpts[6][0]) / 2, scale_x)

  # Start with crop_h = 0 and increase
  crop_h = 0
  target_aspect = target_w / target_h

  def is_left_bound_hit(x_center, crop_w):
    return x_center - crop_w / 2 < 0

  def is_right_bound_hit(x_center, crop_w):
    return x_center + crop_w / 2 > orig_w

  while True:
    crop_h += 2
    crop_w = crop_h * target_aspect

    y0_crop = nose_y - crop_h / 2
    y1_crop = nose_y + crop_h / 2

    left_bound_hit = is_left_bound_hit(x_center, crop_w)
    right_bound_hit = is_right_bound_hit(x_center, crop_w)

    # Check all three conditions
    if y0_crop < 0:
      crop_h -= 2
      break
    if y1_crop > orig_h:
      crop_h -= 2
      break
    if left_bound_hit and right_bound_hit:
      crop_h -= 2
      break

    if left_bound_hit and not is_right_bound_hit(x_center + 1, crop_w):
      x_center += 1
    if right_bound_hit and not is_left_bound_hit(x_center - 1, crop_w):
      x_center -= 1

  crop_w = int(crop_h * target_aspect)
  y0_crop = nose_y - crop_h / 2

  print(f"y0_crop: {y0_crop}, crop_h: {crop_h}, crop_w: {crop_w}")

  print(f"crop: {crop_w / crop_h}, target_aspect: {target_aspect}")

  left = int(x_center - crop_w / 2)
  top = int(y0_crop)

  print(f"left: {left}, top: {top}, crop_w: {crop_w}, crop_h: {crop_h}")

  cropped = image[
    :, :, top : top + crop_h, left : left + crop_w
  ]

  # return [B, C, H, W]
  return cropped
