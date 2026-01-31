RESOLUTIONS = {
  "1280x720 (16:9 Landscape)": (1280, 720),
  "720x1280 (9:16 Portrait)": (720, 1280),
  "480x832 (9:16 Portrait)": (480, 832),
  "832x480 (16:9 Landscape)": (832, 480),
  "512x512 (1:1 Square)": (512, 512),
  "768x768 (1:1 Square)": (768, 768),
  "custom": None,
}


class SR_WanResolution:
  """Resolution selector node with preset options and custom input."""

  CATEGORY = "Simon"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "resolution": (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 Landscape)"}),
        "custom_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
        "custom_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 8}),
      },
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width", "height")
  FUNCTION = "main"

  def main(self, resolution, custom_width, custom_height):
    if resolution == "custom":
      return (custom_width, custom_height)

    width, height = RESOLUTIONS[resolution]
    return (width, height)


NODE_CLASS_MAPPINGS = {"SR_WanResolution": SR_WanResolution}
NODE_DISPLAY_NAME_MAPPINGS = {"SR_WanResolution": SR_WanResolution.NAME}
