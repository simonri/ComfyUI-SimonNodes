import os
import folder_paths
from ultralytics import YOLO
import requests

ultralytics_models_dir = os.path.join(folder_paths.models_dir, "ultralytics")
os.makedirs(ultralytics_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("ultralytics", ultralytics_models_dir)


class SR_UltralyticsModelLoader:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {"model_name": (["yolo11x-pose.pt"], {"tooltip": "These models are loaded from 'ComfyUI/models/ultralytics'"})},
    }

  RETURN_TYPES = ("ULTRALYTICS_MODEL",)
  RETURN_NAMES = ("ultralytics_model",)
  FUNCTION = "load_model"
  CATEGORY = "Simon"

  def __init__(self):
    self.loaded_models = {}

  def load_model(self, model_name: str):
    if model_name in self.loaded_models:
      return (self.loaded_models[model_name],)

    model_path = os.path.join(folder_paths.models_dir, "ultralytics", model_name)
    model_url = f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{model_name}"

    print(model_path)

    if os.path.exists(model_path):
      print("Loading model")
    else:
      response = requests.get(model_url)
      response.raise_for_status()

      with open(model_path, "wb") as f:
        f.write(response.content)

      print("Model downloaded")

    model = YOLO(model_path)
    self.loaded_models[model_name] = model
    return (model,)


NODE_CLASS_MAPPINGS = {
  "SR_UltralyticsModelLoader": SR_UltralyticsModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "SR_UltralyticsModelLoader": "SR Ultralytics Model Loader",
}
