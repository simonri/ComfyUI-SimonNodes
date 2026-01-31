from .hot_reload import setup

setup()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_nodes(module_path: str, name: str, optional: bool = False) -> None:
  """Import and register nodes from a module."""
  try:
    import importlib

    module = importlib.import_module(module_path, package=__package__)
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
  except Exception as e:
    if optional:
      print(f"WanVideoWrapper WARNING: {name} nodes not available: {e}")
    else:
      raise


# AI NODES
register_nodes(".src.nodes.ai.SR_UltralyticsModelLoader", "SR Ultralytics Model Loader")
register_nodes(".src.nodes.ai.SR_UltralyticsCrop", "SR Ultralytics Crop")

# IMAGE NODES
register_nodes(".src.nodes.image.SR_SeedVR_Upscale", "SR SeedVR Upscale")
register_nodes(".src.nodes.image.SR_ImageResize", "SR Image Resize")

# VIDEO NODES
register_nodes(".src.nodes.video.SR_RIFE", "SR RIFE")

# UTILITY NODES
register_nodes(".src.nodes.utility.SR_Seed", "SR Seed")

# WAN NODES
register_nodes(".src.nodes.wan.SR_WanSampler", "SR Wan Sampler")
register_nodes(".src.nodes.wan.SR_WanResolution", "SR Wan Resolution")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
