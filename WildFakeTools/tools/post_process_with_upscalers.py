from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
from .base_tool import BaseTool


class PostProcessWithUpscalersTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="post_process_with_upscalers",
            description="Enhances image resolution while preserving realistic textures, making the image appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        upscale_factor = params.get("upscale_factor", 2.0)
        preserve_textures = params.get("preserve_textures", True)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        original_size = image.size

        # Convert image to numpy array
        img_array = np.array(image)

        # Apply upscaling
        if preserve_textures:
            # Cubic zoom, preserving texture.
            zoom_factors = (upscale_factor, upscale_factor, 1)  # Keep the color channels intact
            upscaled_array = zoom(img_array, zoom_factors, order=3)  # Cubic interpolation
        else:
            # Nearest-neighbour block replication.
            upscaled_array = np.kron(img_array, np.ones((int(upscale_factor), int(upscale_factor), 1)))

        # Clip values to be in valid range
        upscaled_array = np.clip(upscaled_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        upscaled_image = Image.fromarray(upscaled_array)

        # Resize back to original size if needed
        upscaled_image = upscaled_image.resize(original_size, Image.LANCZOS)

        return upscaled_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "upscale_factor": {
                "type": "float",
                "min": 1.5,
                "max": 4.0,
                "description": "Factor by which to upscale the image.",
                "default": 2.0,
                "required": False
            },
            "preserve_textures": {
                "type": "bool",
                "description": "Whether to preserve realistic textures during upscaling.",
                "default": True,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Upscale factor between 1.5 and 4.0; use when enhancing AI-generated images for realism."