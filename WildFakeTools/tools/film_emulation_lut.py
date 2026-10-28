from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class FilmEmulationLutTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="film_emulation_lut",
            description="Applies a film-stock LUT to remap the image's color palette, emulating real film aesthetics."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        intensity = params.get("intensity", 1.0)
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image) / 255.0  # Normalize to [0, 1]

        # Create a simple film emulation LUT (for demonstration purposes)
        # This LUT will adjust colors to mimic a film-like aesthetic
        lut = np.array([
            [0.9, 0.8, 0.7],  # Red channel adjustment
            [0.8, 0.9, 0.7],  # Green channel adjustment
            [0.7, 0.8, 0.9]   # Blue channel adjustment
        ])

        # Apply the LUT
        for i in range(3):  # For each color channel
            img_array[..., i] = np.clip(img_array[..., i] * lut[i, i] * intensity, 0, 1)

        # Convert back to image
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "intensity": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "description": "Intensity of the LUT effect, where 1.0 is normal strength.",
                "default": 1.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0.0 to 2.0; useful for enhancing AI-generated images to appear more like film photographs."