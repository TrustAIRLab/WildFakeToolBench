from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class MultiPassUpscaleDenoiseTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="multi_pass_upscale_denoise",
            description="This tool enhances image realism by first upscaling the image and then applying a denoising filter to refine textures and reduce artifacts."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        upscale_factor = params.get("upscale_factor", 2)
        denoise_strength = params.get("denoise_strength", 0.5)
        
        # Convert image to RGB if not already
        image = image.convert("RGB")

        # Upscale the image, capped at an absolute maximum output size.
        MAX_DIMENSION = 2048
        target_size = (image.width * upscale_factor, image.height * upscale_factor)
        if max(target_size) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(target_size)
            target_size = (max(1, int(target_size[0] * scale)), max(1, int(target_size[1] * scale)))
        new_size = target_size
        upscaled_image = image.resize(new_size, Image.LANCZOS)
        
        # Apply denoising filter
        denoise_radius = 1 + denoise_strength * 2  # Scale radius based on denoise_strength
        denoised_image = upscaled_image.filter(ImageFilter.GaussianBlur(radius=denoise_radius))
        
        return denoised_image

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
            "denoise_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the denoising effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use upscale_factor between 2 and 4, and denoise_strength between 0.0 and 1.0 for optimal results."