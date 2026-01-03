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


register_nodes(".src.nodes.ultralytics_model_loader", "Ultralytics Model Loader")
register_nodes(".src.nodes.ultralytics_crop", "Ultralytics Crop")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
