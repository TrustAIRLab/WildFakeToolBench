from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class PhotographicImperfectionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="photographic_imperfections",
            description="Adds photographic imperfections like lens vignetting and chromatic aberration to an image, making it appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        vignetting_strength = params.get("vignetting_strength", 0.5)
        chromatic_aberration_strength = params.get("chromatic_aberration_strength", 0.5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Apply vignetting
        height, width = img_array.shape[:2]
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vignette_mask = 1 - (distance / max_radius) * vignetting_strength
        vignette_mask = np.clip(vignette_mask, 0, 1)

        for i in range(3):  # Apply to each channel
            img_array[..., i] = img_array[..., i] * vignette_mask

        # Apply chromatic aberration
        aberration_offset = int(chromatic_aberration_strength * 10)  # Control the offset
        red_channel = np.roll(img_array[..., 0], aberration_offset, axis=1)
        blue_channel = np.roll(img_array[..., 2], -aberration_offset, axis=1)
        img_array[..., 0] = red_channel
        img_array[..., 2] = blue_channel

        # Clip values to be in valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "vignetting_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the vignetting effect.",
                "default": 0.5,
                "required": False
            },
            "chromatic_aberration_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the chromatic aberration effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters range from 0 to 1; use to add realistic imperfections to AI-generated images."