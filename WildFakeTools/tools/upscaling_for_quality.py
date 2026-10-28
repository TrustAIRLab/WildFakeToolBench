from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class UpscalingForQualityTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="upscaling_for_quality",
            description="Upscaling enhances the detail in images, making them more visually appealing and closer to the quality of real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        upscale_factor = params.get("upscale_factor", 2)
        algorithm = params.get("algorithm", "bicubic")

        # Convert image to RGB if not already
        image = image.convert("RGB")

        # Upscale the image, capped at an absolute maximum output size.
        MAX_DIMENSION = 2048
        target_size = (image.width * upscale_factor, image.height * upscale_factor)
        if max(target_size) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(target_size)
            target_size = (max(1, int(target_size[0] * scale)), max(1, int(target_size[1] * scale)))

        # Upscale the image using the specified algorithm
        if algorithm == "nearest-neighbor":
            resampled_image = image.resize(target_size, Image.Resampling.NEAREST)
        elif algorithm == "bicubic":
            resampled_image = image.resize(target_size, Image.Resampling.BICUBIC)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        return resampled_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "upscale_factor": {
                "type": "int",
                "min": 2,
                "max": 4,
                "description": "Factor by which to upscale the image.",
                "default": 2,
                "required": False
            },
            "algorithm": {
                "type": "str",
                "min": None,
                "max": None,
                "description": "Upscaling algorithm to use ('nearest-neighbor' or 'bicubic').",
                "default": "bicubic",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Upscale factor: 2-4; Algorithm: 'nearest-neighbor' or 'bicubic'. Use for enhancing AI-generated images."