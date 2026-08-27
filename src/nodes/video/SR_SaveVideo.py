import os
import sys
import re
import json
import shutil
import datetime
import subprocess

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.utils import ProgressBar

ENCODE_ARGS = ("utf-8", "backslashreplace")

# Fixed encode target — h264/mp4, the settings VideoHelperSuite's h264-mp4.json preset
# ships (crf 19, yuv420p, BT.709 tagged). No format dropdown: this is the one format
# every downstream consumer (ffmpeg re-encodes, browsers, Telegram) can rely on.
EXTENSION = "mp4"
PIX_FMT = "yuv420p"
CRF = "19"
FAKE_TRC = "bt709"
AUDIO_PASS = ["-c:a", "aac", "-movflags", "use_metadata_tags"]
DIM_ALIGNMENT = 2  # yuv420p needs even width/height


def _get_ffmpeg_path():
  try:
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()
  except Exception:
    return shutil.which("ffmpeg")


FFMPEG_PATH = _get_ffmpeg_path()


def _merge_filter_args(args, ftype="-vf"):
  try:
    start_index = args.index(ftype) + 1
    index = start_index
    while True:
      index = args.index(ftype, index)
      args[start_index] += "," + args[index + 1]
      args.pop(index)
      args.pop(index)
  except ValueError:
    pass


def _tensor_to_bytes(tensor):
  tensor = tensor.cpu().numpy() * 255 + 0.5
  return np.clip(tensor, 0, 255).astype(np.uint8)


def _to_pingpong(inp):
  if not hasattr(inp, "__getitem__"):
    inp = list(inp)
  yield from inp
  for i in range(len(inp) - 2, 0, -1):
    yield inp[i]


def _build_video_filter(loop_count, num_frames, grain_strength, width, height):
  """Builds the -filter_complex graph for the main encode: fixed cleanup/look passes
  plus optional grain, always ending in [vout]. Order follows the standard grading
  chain (deband/dither, then deflicker, then grade/look, then scale).

  No chromatic aberration: rgbashift applies a *uniform* shift across the whole
  frame, but real lens CA is radial — near-zero at the center, growing toward the
  corners. A flat shift is wrong in exactly the place that matters most: it fringes
  whatever's in the middle of the frame, which for this content is almost always
  the subject. Tested on a real photo at rgbashift's minimum non-zero shift (1px)
  and it was clearly visible on the subject's arm against the sky — not a "did
  budget get spent here" cue, just a visible artifact. Doing this properly would
  need a radial (distance-from-center) shift, which is future work if wanted, not
  a tunable knob on this one.

  - deband: smooths 8-bit gradient banding before anything else can bake it in further.
  - deflicker: temporal luminance smoothing — the actual fix for causal-VAE flicker,
    as opposed to grain, which only masks it. Frame-count window, not scaled by fps —
    the instability it corrects is a property of the generated frame sequence itself,
    not of how fast it's later tagged to play back.
  - vignette: mild corner falloff. `aspect` MUST be set to the frame's own w/h —
    the filter's default (1/1, i.e. square) makes it wildly asymmetric on anything
    else: measured 36% top/bottom darkening vs 12% left/right on a 720x1280 portrait
    frame with the default. `angle` was retuned smaller after fixing aspect, since
    the old value plus the aspect fix would otherwise have gotten stronger, not weaker.
  - halation: highlights above a threshold, blurred into a soft glow, blended back in —
    approximates the light-scatter halo real film stock shows around bright areas.
    Blur radius scaled to frame size so it stays a consistent glow size relative to
    the frame instead of a fixed-pixel blob.
  - grain (optional): split/blend sub-graph blending noise in more heavily where the
    source is dark, temporally decorrelated per frame — that branching is why this
    can't just be a plain -vf chain. Strength tuned down from an earlier pass (24)
    to 8 by default after testing against real photographic content, not just
    synthetic gradients — 24 turned a fine floral dress pattern into visible static;
    gradient/noise-ratio tests alone didn't catch that because they lack the
    high-frequency detail real content has.
  """
  min_dim = min(width, height)
  halo_sigma = max(2, round(min_dim * 6 / 256))  # matches the 256px-wide tuning baseline

  prefix = ["format=yuv420p"]
  if loop_count > 0:
    prefix.append(f"loop=loop={loop_count}:size={num_frames}")
  prefix.append("deband=1thr=0.02:2thr=0.02:3thr=0.02")
  prefix.append("deflicker=mode=pm:size=6")
  prefix.append(f"vignette=angle=PI/8:aspect={width}/{height}")
  prefix_str = ",".join(prefix)

  graded = (
    f"[0:v]{prefix_str},split=2[grade][htmp];"
    # lutyuv, not geq: this expression only ever reads the pixel's own value (no
    # neighbor/coordinate reference), so it's a pure per-value lookup — geq
    # evaluates the expression interpreted, per pixel, per frame, which is far
    # more expensive for something a 256-entry LUT already covers exactly.
    f"[htmp]lutyuv=y='if(gt(val,180),val-180,0)':u=128:v=128,gblur=sigma={halo_sigma}[halo];"
    # lut2, not blend: U/V are centered at 128 ("no color"), and blend's
    # all_mode=addition adds two 128-centered planes together, pushing the whole
    # frame magenta. lut2 gives explicit per-plane control instead: c0 (luma)
    # gets the glow added (x=grade, y=halo), c1/c2 (chroma) pass [grade] through
    # untouched via plain 'x'.
    "[grade][halo]lut2=c0='clip(x+y*0.4,0,255)':c1='x':c2='x'[lit]"
  )

  if grain_strength > 0:
    return (
      f"{graded};"
      "[lit]split=3[base][ntmp][mtmp];"
      f"[ntmp]noise=c0s={grain_strength}:c0f=t+u[grain];"
      "[mtmp]lutyuv=y=negval[mask];"
      "[base][grain][mask]maskedmerge[vout]"
    )
  return f"{graded};[lit]copy[vout]"


def _ffmpeg_process(args, save_metadata, video_metadata, file_path, env):
  """Pipe raw frames into ffmpeg, embedding prompt/workflow metadata first when
  requested. Yields the total frame count once the pipe closes."""
  res = None
  frame_data = yield
  total_frames_output = 0

  if save_metadata:
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")

    def escape_ffmpeg_metadata(key, value):
      value = str(value)
      for old, new in (("\\", "\\\\"), (";", "\\;"), ("#", "\\#"), ("=", "\\="), ("\n", "\\\n")):
        value = value.replace(old, new)
      return f"{key}={value}"

    with open(metadata_path, "w") as f:
      f.write(";FFMETADATA1\n")
      for key in ("prompt", "workflow"):
        if key in video_metadata:
          f.write(escape_ffmpeg_metadata(key, json.dumps(video_metadata[key])) + "\n")
      for k, v in video_metadata.items():
        if k not in ("prompt", "workflow"):
          f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")

    # The metadata file must be appended *after* our raw-video input (not
    # prepended) so it doesn't shift the video input to index 1 — the
    # filter_complex graph in `args` references it explicitly as "0:v". That
    # makes it input 1, so -map_metadata must say so explicitly too: ffmpeg's
    # default is to pull global metadata from input 0, which is the rawvideo
    # pipe (nothing to inherit) now that it isn't input 0 anymore.
    in_args_len = args.index("-i") + 2
    m_args = (
      args[:in_args_len]
      + ["-i", metadata_path]
      + args[in_args_len:]
      + ["-map_metadata", "1", "-metadata", "creation_time=now", "-movflags", "use_metadata_tags"]
    )
    with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
      try:
        while frame_data is not None:
          proc.stdin.write(frame_data)
          frame_data = yield
          total_frames_output += 1
        proc.stdin.flush()
        proc.stdin.close()
        res = proc.stderr.read()
      except BrokenPipeError:
        err = proc.stderr.read()
        if os.path.exists(file_path):
          raise Exception("An error occurred in the ffmpeg subprocess:\n" + err.decode(*ENCODE_ARGS))
        print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)

  # `is None`, not a truthiness/equality check against b"": save_metadata=False
  # never enters the block above, so res is genuinely None and this is the only
  # encode. But if save_metadata=True and the metadata pass merely printed a
  # non-fatal warning to stderr (res == some non-empty bytes, not an exception),
  # `!= b""` used to treat that as "the first pass didn't produce output" and
  # retry here — except frame_data is already exhausted (0 frames to write) and
  # `-n` now refuses to overwrite the file the first pass already wrote.
  if res is None:
    with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env) as proc:
      try:
        while frame_data is not None:
          proc.stdin.write(frame_data)
          frame_data = yield
          total_frames_output += 1
        proc.stdin.flush()
        proc.stdin.close()
        res = proc.stderr.read()
      except BrokenPipeError:
        res = proc.stderr.read()
        raise Exception("An error occurred in the ffmpeg subprocess:\n" + res.decode(*ENCODE_ARGS))

  yield total_frames_output
  if res and len(res) > 0:
    print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


class SR_SaveVideo:
  """
  Trimmed-down Video Combine: pipes a frame batch straight through ffmpeg into an
  h264/mp4 file. No format dropdown, no Pillow gif/webp branch, no VAE-latent
  decode, no batch manager. Also trims the first/last few frames (causal video
  VAEs are least stable at the edges of their latent window) and can blend in
  temporally-decorrelated, shadow-weighted grain to mask VAE banding/flicker.
  """

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "images": ("IMAGE",),
        "frame_rate": ("FLOAT", {"default": 8, "min": 1, "step": 1}),
        "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
        "filename_prefix": ("STRING", {"default": "SR_Video"}),
        "trim_frames": (
          "INT",
          {"default": 2, "min": 0, "max": 30, "tooltip": "Frames dropped from each end (causal VAEs are weakest at the edges of their latent window)."},
        ),
        "grain_strength": (
          "INT",
          {"default": 8, "min": 0, "max": 100, "tooltip": "Temporal, shadow-weighted grain blended in to mask VAE banding/flicker. 0 disables it."},
        ),
        "pingpong": ("BOOLEAN", {"default": False}),
        "save_output": ("BOOLEAN", {"default": True}),
      },
      "optional": {
        "audio": ("AUDIO",),
      },
      "hidden": {
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
      },
    }

  RETURN_TYPES = ("VHS_FILENAMES",)
  RETURN_NAMES = ("filenames",)
  OUTPUT_NODE = True
  CATEGORY = "Simon"
  FUNCTION = "save_video"

  def save_video(
    self,
    images,
    frame_rate,
    loop_count,
    filename_prefix,
    trim_frames,
    grain_strength,
    pingpong,
    save_output,
    audio=None,
    prompt=None,
    extra_pnginfo=None,
  ):
    if FFMPEG_PATH is None:
      raise ProcessLookupError(
        "ffmpeg is required for SR Save Video and could not be found. Install imageio-ffmpeg "
        "or place an ffmpeg executable on the system PATH."
      )
    if images is None or images.size(0) == 0:
      return ((save_output, []),)

    if trim_frames > 0 and images.size(0) - 2 * trim_frames >= 1:
      images = images[trim_frames : images.size(0) - trim_frames]

    num_frames = images.size(0)
    pbar = ProgressBar(num_frames)
    first_image = images[0]
    has_alpha = first_image.shape[-1] == 4
    frames = iter(images)

    output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)

    metadata = PngInfo()
    video_metadata = {}
    if prompt is not None:
      metadata.add_text("prompt", json.dumps(prompt))
      video_metadata["prompt"] = prompt
    if extra_pnginfo is not None:
      for k, v in extra_pnginfo.items():
        metadata.add_text(k, json.dumps(v))
        video_metadata[k] = v
    metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

    max_counter = 0
    matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
    for existing_file in os.listdir(full_output_folder):
      match = matcher.fullmatch(existing_file)
      if match:
        max_counter = max(max_counter, int(match.group(1)))
    counter = max_counter + 1

    output_files = []
    first_image_file = f"{filename}_{counter:05}.png"
    first_image_path = os.path.join(full_output_folder, first_image_file)
    Image.fromarray(_tensor_to_bytes(first_image)).save(first_image_path, pnginfo=metadata, compress_level=4)
    output_files.append(first_image_path)

    if (first_image.shape[1] % DIM_ALIGNMENT) or (first_image.shape[0] % DIM_ALIGNMENT):
      to_pad = (-first_image.shape[1] % DIM_ALIGNMENT, -first_image.shape[0] % DIM_ALIGNMENT)
      padding = (to_pad[0] // 2, to_pad[0] - to_pad[0] // 2, to_pad[1] // 2, to_pad[1] - to_pad[1] // 2)
      padfunc = torch.nn.ReplicationPad2d(padding)

      def pad(image):
        image = image.permute((2, 0, 1))  # HWC to CHW
        padded = padfunc(image.to(dtype=torch.float32))
        return padded.permute((1, 2, 0))

      frames = map(pad, frames)
      dimensions = (
        -first_image.shape[1] % DIM_ALIGNMENT + first_image.shape[1],
        -first_image.shape[0] % DIM_ALIGNMENT + first_image.shape[0],
      )
    else:
      dimensions = (first_image.shape[1], first_image.shape[0])

    # loop's -filter_complex window (below) must match what's actually piped in,
    # not the pre-pingpong count, or it only loops the forward half of each cycle.
    piped_frame_count = num_frames
    if pingpong:
      frames = _to_pingpong(frames)
      if num_frames > 2:
        piped_frame_count = num_frames + (num_frames - 2)
        pbar.total = piped_frame_count

    frames = map(_tensor_to_bytes, frames)
    i_pix_fmt = "rgba" if has_alpha else "rgb24"

    file = f"{filename}_{counter:05}.{EXTENSION}"
    file_path = os.path.join(full_output_folder, file)

    args = [
      FFMPEG_PATH, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
      # Input is undefined-generic-RGB (sRGB in practice). Tell ffmpeg it's already
      # BT.709 so it doesn't silently reinterpret colors during the yuv420p convert.
      "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
      "-color_trc", FAKE_TRC,
      "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-",
      "-filter_complex", _build_video_filter(loop_count, piped_frame_count, grain_strength, dimensions[0], dimensions[1]), "-map", "[vout]",
      "-n", "-c:v", "libx264", "-pix_fmt", PIX_FMT, "-crf", CRF,
      "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    ]

    frame_bytes = map(lambda x: x.tobytes(), frames)
    env = os.environ.copy()

    output_process = _ffmpeg_process(args, True, video_metadata, file_path, env)
    output_process.send(None)

    for frame in frame_bytes:
      pbar.update(1)
      output_process.send(frame)

    total_frames_output = num_frames
    try:
      total_frames_output = output_process.send(None)
      output_process.send(None)
    except StopIteration:
      pass

    output_files.append(file_path)

    a_waveform = audio["waveform"] if audio is not None and "waveform" in audio else None
    if a_waveform is not None:
      output_file_with_audio = f"{filename}_{counter:05}-audio.{EXTENSION}"
      output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
      channels = a_waveform.size(1)
      if trim_frames > 0:
        # Video's frame 0 is now the original's frame `trim_frames` — audio must
        # drop the same span or it leads the picture by trim_frames/frame_rate
        # seconds (83ms at the defaults, well past the ~45ms lip-sync detection
        # threshold). -shortest handles the trailing edge; only the start needs
        # correcting here.
        audio_offset_samples = int(round(trim_frames / frame_rate * audio["sample_rate"]))
        a_waveform = a_waveform[..., audio_offset_samples:]
      # total_frames_output counts frames piped in, before ffmpeg's loop filter
      # (if any) repeats that window internally -- scale by (loop_count+1) to
      # match the actual output duration, or -shortest truncates the looped
      # video down to the un-looped audio length instead of the other way around.
      min_audio_dur = total_frames_output * (loop_count + 1) / frame_rate + 1
      apad = ["-af", f"apad=whole_dur={min_audio_dur}"]
      mux_args = (
        [
          FFMPEG_PATH, "-v", "error", "-n", "-i", file_path,
          "-ar", str(audio["sample_rate"]), "-ac", str(channels),
          "-f", "f32le", "-i", "-", "-c:v", "copy",
        ]
        + AUDIO_PASS
        + apad
        + ["-shortest", output_file_with_audio_path]
      )
      audio_data = a_waveform.squeeze(0).transpose(0, 1).numpy().tobytes()
      _merge_filter_args(mux_args, "-af")
      try:
        res = subprocess.run(mux_args, input=audio_data, env=env, capture_output=True, check=True)
      except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmpeg audio-mux subprocess:\n" + e.stderr.decode(*ENCODE_ARGS))
      if res.stderr:
        print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
      output_files.append(output_file_with_audio_path)
      file = output_file_with_audio

    # "images" + "animated": (True,) is ComfyUI core's own preview convention (used by
    # its native SaveVideo/SaveWEBP/SaveAnimatedPNG nodes, see comfy_api/latest/_ui.py's
    # PreviewVideo/SavedImages) — it renders a video/animated-image widget for any node,
    # unlike VHS's "gifs" key, which its web/js only attaches to nodes literally named
    # VHS_VideoCombine.
    preview = {
      "filename": file,
      "subfolder": subfolder,
      "type": "output" if save_output else "temp",
    }
    return {"ui": {"images": [preview], "animated": (True,)}, "result": ((save_output, output_files),)}


NODE_CLASS_MAPPINGS = {"SR_SaveVideo": SR_SaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SR_SaveVideo": "SR Save Video"}
