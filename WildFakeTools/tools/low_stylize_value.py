from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class LowStylizeValueTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="low_stylize_value",
            description="This tool applies a low stylization effect to enhance realism in images by reducing artistic distortions."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        stylize_value = params.get("stylize_value", 100)  # Default stylize value

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create a blurred version of the image to simulate low stylization
        blur_radius = (200 - stylize_value) / 100  # Map stylize value to blur radius
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Blend the original image with the blurred image to maintain some details
        blended_image = Image.blend(image, blurred_image, alpha=0.5)

        return blended_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "stylize_value": {
                "type": "int",
                "min": 75,
                "max": 200,
                "description": "Stylization level to apply, lower values enhance realism.",
                "default": 100,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Stylize value ranges from 75 to 200; lower values enhance realism, useful for reducing artistic effects."