from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class VignetteApplicationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="vignette_application",
            description="Apply a vignette effect to enhance the film-like quality of the image. This effect draws focus to the center, mimicking traditional photography techniques."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        strength = params.get("strength", 50) / 100.0  # Convert to a fraction
        brightness = params.get("brightness", 50) / 100.0  # Convert to a fraction
        
        # Convert image to RGB
        image = image.convert("RGB")
        width, height = image.size
        vignette = np.zeros((height, width), dtype=np.float32)

        # Create a vignette mask
        for y in range(height):
            for x in range(width):
                # Calculate the distance from the center
                dx = (x - width / 2) / (width / 2)
                dy = (y - height / 2) / (height / 2)
                distance = np.sqrt(dx * dx + dy * dy)
                vignette[y, x] = max(0, 1 - distance)

        # Scale the vignette effect
        vignette = 1 - (vignette * strength)
        vignette = vignette * (1 - brightness) + brightness  # Adjust brightness

        # Apply the vignette effect to each channel
        image_array = np.array(image, dtype=np.float32) / 255.0
        for i in range(3):  # For R, G, B channels
            image_array[..., i] *= vignette

        # Convert back to PIL Image
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0,
                "max": 100,
                "description": "The strength of the vignette effect (0-100%).",
                "default": 50,
                "required": False
            },
            "brightness": {
                "type": "float",
                "min": 0,
                "max": 100,
                "description": "The brightness adjustment for the vignette (0-100%).",
                "default": 50,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength and brightness should be set between 0 and 100 for optimal vignette effects."