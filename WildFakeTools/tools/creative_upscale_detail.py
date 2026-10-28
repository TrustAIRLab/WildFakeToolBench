from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class CreativeUpscaleDetailTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="creative_upscale_detail",
            description="Enhances the realism of an image by applying a creative upscaling process that adds fine details and textures, mimicking the micro-details found in real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        upscale_factor = params.get("upscale_factor", 2)
        detail_strength = params.get("detail_strength", 0.5)

        # Convert image to RGB
        image = image.convert("RGB")

        # Upscale the image, capped at an absolute maximum output size.
        MAX_DIMENSION = 2048
        target_size = (image.width * upscale_factor, image.height * upscale_factor)
        if max(target_size) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(target_size)
            target_size = (max(1, int(target_size[0] * scale)), max(1, int(target_size[1] * scale)))

        # Nearest-neighbour upscale (same replication semantics as the previous
        # per-pixel Python loop, but bounded and orders of magnitude faster).
        upscaled_image = np.array(image.resize(target_size, Image.NEAREST))

        # Apply Gaussian filter to add detail
        if detail_strength > 0:
            detail_layer = gaussian_filter(upscaled_image, sigma=(1, 1, 0))
            upscaled_image = np.clip(upscaled_image + detail_strength * (upscaled_image - detail_layer), 0, 255).astype(np.uint8)

        return Image.fromarray(upscaled_image, mode="RGB")

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
            "detail_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the detail enhancement.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use upscale_factor between 2 and 4, and detail_strength between 0.0 and 1.0 for enhancing image realism."