from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class LightSourceDescriptionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="light_source_description",
            description="Enhances the realism of an image by simulating light source characteristics such as direction, quality, and color temperature."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Extract parameters
        light_direction = params.get("light_direction", "top")
        light_quality = params.get("light_quality", "soft")
        color_temperature = params.get("color_temperature", 5000)

        # Simulate light direction and quality
        if light_direction == "top":
            light_effect = np.array([1.0, 1.0, 1.0])  # Top light is neutral
        elif light_direction == "side":
            light_effect = np.array([1.2, 1.0, 0.8])  # Side light adds warmth
        else:  # bottom
            light_effect = np.array([0.8, 0.8, 1.0])  # Bottom light is cooler

        # Adjust quality
        if light_quality == "soft":
            img_array = ImageEnhance.Brightness(image).enhance(1.1)  # Soft light increases brightness slightly
        else:  # hard
            img_array = ImageEnhance.Contrast(image).enhance(1.2)  # Hard light increases contrast

        # Adjust color temperature
        if color_temperature < 3000:
            img_array = img_array * np.array([1.0, 0.8, 0.6])  # Warm light
        elif color_temperature > 6000:
            img_array = img_array * np.array([0.6, 0.7, 1.0])  # Cool light

        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "light_direction": {
                "type": "str",
                "description": "Direction of the light source (top, side, bottom).",
                "default": "top",
                "required": False
            },
            "light_quality": {
                "type": "str",
                "description": "Quality of the light (soft, hard).",
                "default": "soft",
                "required": False
            },
            "color_temperature": {
                "type": "int",
                "min": 1000,
                "max": 10000,
                "description": "Color temperature of the light source in Kelvin.",
                "default": 5000,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: light_direction (top/side/bottom), light_quality (soft/hard), color_temperature (1000-10000). Use to enhance realism."