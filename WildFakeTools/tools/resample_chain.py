from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class ResampleChainTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="resample_chain",
            description="This tool applies a series of resampling operations including upsampling, low-pass filtering, and downsampling to alter the pixel distribution of AI-generated images, making them resemble real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        upsample_factor = params.get("upsample_factor", 2.0)
        low_pass_cutoff = params.get("low_pass_cutoff", 1.0)
        downsample_factor = params.get("downsample_factor", 2.0)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        orig_size = image.size  # (W, H)

        # Upsampling — PIL resize handles arbitrary (non-integer) factors
        # correctly, unlike a manual strided-array fill.
        up_size = (max(1, round(orig_size[0] * upsample_factor)), max(1, round(orig_size[1] * upsample_factor)))
        upsampled_image = np.array(image.resize(up_size, Image.BICUBIC))

        # Low-pass filtering
        sigma = low_pass_cutoff / 2.0
        filtered_image = gaussian_filter(upsampled_image, sigma=(sigma, sigma, 0))

        # Downsampling back toward the target size
        down_size = (max(1, round(up_size[0] / downsample_factor)), max(1, round(up_size[1] / downsample_factor)))
        downsampled_image = Image.fromarray(filtered_image.astype(np.uint8), mode="RGB").resize(down_size, Image.BICUBIC)

        return downsampled_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "upsample_factor": {
                "type": "float",
                "min": 1.0,
                "max": 5.0,
                "description": "Factor by which to upsample the image.",
                "default": 2.0,
                "required": False
            },
            "low_pass_cutoff": {
                "type": "float",
                "min": 0.1,
                "max": 5.0,
                "description": "Cutoff frequency for the low-pass filter.",
                "default": 1.0,
                "required": False
            },
            "downsample_factor": {
                "type": "float",
                "min": 1.0,
                "max": 5.0,
                "description": "Factor by which to downsample the image.",
                "default": 2.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters range: upsample_factor (1.0 to 5.0), low_pass_cutoff (0.1 to 5.0), downsample_factor (1.0 to 5.0). Use this tool to enhance AI-generated images to appear more like real photographs."