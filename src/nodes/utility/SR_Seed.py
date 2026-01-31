import random
import logging
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

# Some extensions set a global seed, making server-generated seeds non-random.
# We maintain our own random state to ensure true randomness.
_initial_state = random.getstate()
random.seed(datetime.now().timestamp())
_seed_random_state = random.getstate()
random.setstate(_initial_state)
del _initial_state

MAX_SEED = 1125899906842624  # 2^50


@contextmanager
def _isolated_random():
  """Context manager that uses our isolated random state without affecting global state."""
  global _seed_random_state
  prev_state = random.getstate()
  random.setstate(_seed_random_state)
  try:
    yield
  finally:
    _seed_random_state = random.getstate()
    random.setstate(prev_state)


def new_random_seed():
  """Generate a new random seed using isolated random state."""
  with _isolated_random():
    return random.randint(1, MAX_SEED)


class SR_Seed:
  """Seed node that provides consistent seed handling for workflows."""

  NAME = "SR Seed"
  CATEGORY = "Simon"

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "seed": (
          "INT",
          {
            "default": 0,
            "min": -MAX_SEED,
            "max": MAX_SEED,
          },
        ),
      },
      "hidden": {
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
        "unique_id": "UNIQUE_ID",
      },
    }

  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("SEED",)
  FUNCTION = "main"

  @classmethod
  def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None):
    """Return varying value for special seeds to prevent caching."""
    if seed in (-1, -2, -3):
      return new_random_seed()
    return seed

  def main(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
    """Returns the seed, generating a random one if a special value is passed."""
    if seed not in (-1, -2, -3):
      return (seed,)

    # Handle special seed values (typically from API calls)
    logger.warning(f'Received special seed "{seed}". This is unexpected from the ComfyUI frontend.')
    if seed in (-2, -3):
      action = "increment" if seed == -2 else "decrement"
      logger.warning(f"Cannot {action} seed from server; generating random seed instead.")

    original_seed = seed
    seed = new_random_seed()
    logger.info(f"Generated random seed: {seed}")

    # Save generated seed to metadata
    self._update_workflow_metadata(unique_id, extra_pnginfo, original_seed, seed)
    self._update_prompt_metadata(unique_id, prompt, seed)

    return (seed,)

  def _update_workflow_metadata(self, unique_id, extra_pnginfo, original_seed, new_seed):
    """Update the workflow metadata with the generated seed."""
    if unique_id is None:
      logger.warning("Cannot save seed to metadata: node ID not provided.")
      return

    if extra_pnginfo is None:
      logger.warning("Cannot save seed to workflow metadata: workflow not provided.")
      return

    workflow_node = next((x for x in extra_pnginfo["workflow"]["nodes"] if str(x["id"]) == str(unique_id)), None)
    if workflow_node is None or "widgets_values" not in workflow_node:
      logger.warning("Cannot save seed to workflow metadata: node not found.")
      return

    for index, widget_value in enumerate(workflow_node["widgets_values"]):
      if widget_value == original_seed:
        workflow_node["widgets_values"][index] = new_seed
        break

  def _update_prompt_metadata(self, unique_id, prompt, new_seed):
    """Update the prompt metadata with the generated seed."""
    if unique_id is None:
      return

    if prompt is None:
      logger.warning("Cannot save seed to prompt metadata: prompt not provided.")
      return

    prompt_node = prompt.get(str(unique_id))
    if prompt_node is None or "inputs" not in prompt_node or "seed" not in prompt_node["inputs"]:
      logger.warning("Cannot save seed to prompt metadata: node not found.")
      return

    prompt_node["inputs"]["seed"] = new_seed


NODE_CLASS_MAPPINGS = {"SR_Seed": SR_Seed}
NODE_DISPLAY_NAME_MAPPINGS = {"SR_Seed": SR_Seed.NAME}
