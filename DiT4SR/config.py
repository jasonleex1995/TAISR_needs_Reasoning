# Weight paths for the DiT4SR backbone. Edit these (or set the matching env
# vars) to point at your local downloads. See README for download links.
#
#   SD35_PATH      : Stability AI Stable Diffusion 3.5 Medium
#                    (https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
#   DIT4SR_Q_PATH  : DiT4SR-Q weights (https://github.com/Adam-duan/DiT4SR)
import os

SD35_PATH = os.environ.get("SD35_PATH", "/path/to/stable-diffusion-3.5-medium")
DIT4SR_Q_PATH = os.environ.get("DIT4SR_Q_PATH", "/path/to/DiT4SR/dit4sr_q")
