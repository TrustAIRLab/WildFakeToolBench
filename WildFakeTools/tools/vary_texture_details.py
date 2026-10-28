from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class VaryTextureDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="vary_texture_details",
            description="Adds variability to texture details in AI images to create a more organic look by introducing imperfections typical in real-world photography.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        texture_variability = params.get("texture_variability", 0.5)
        detail_strength = params.get("detail_strength", 0.5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create a noise layer based on the texture variability
        noise = np.random.normal(loc=0, scale=texture_variability * 255, size=img_array.shape)
        noisy_image = img_array + noise

        # Apply Gaussian filter to create a detail effect based on detail strength
        detail_effect = gaussian_filter(noisy_image, sigma=(detail_strength * 5, detail_strength * 5, 0))

        # Clip values to ensure they are within valid range
        detail_effect = np.clip(detail_effect, 0, 255).astype(np.uint8)

        return Image.fromarray(detail_effect, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "texture_variability": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the amount of texture variability added to the image.",
                "default": 0.5,
                "required": False
            },
            "detail_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the detail effect applied to the image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Texture variability (0-1) and detail strength (0-1) control the organic appearance of the image."