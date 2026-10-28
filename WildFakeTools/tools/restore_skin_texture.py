from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class RestoreSkinTextureTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="restore_skin_texture",
            description="Adds high-frequency pore and fine-line texture to AI-generated headshots, enhancing realism by introducing subtle tonal variations and micro-contrast found in real skin."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        texture_strength = params.get("texture_strength", 0.5)
        frequency_range = params.get("frequency_range", (2, 5))

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create a high-frequency texture using Gaussian filter
        blurred = gaussian_filter(img_array, sigma=frequency_range[0])
        high_freq_texture = img_array - blurred

        # Scale the high-frequency texture by the texture strength
        textured_image = img_array + (high_freq_texture * texture_strength)
        textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)

        return Image.fromarray(textured_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "texture_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the texture to be added to the image.",
                "default": 0.5,
                "required": False
            },
            "frequency_range": {
                "type": "tuple",
                "min": (1, 1),
                "max": (10, 10),
                "description": "Range of frequencies for the high-frequency texture.",
                "default": (2, 5),
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Texture strength: 0.0 to 1.0; Frequency range: (1, 10) for subtle to strong texture. Best for enhancing AI-generated headshots."