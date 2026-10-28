from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class LightingHarmonizationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="lighting_harmonization",
            description="Adjusts the lighting direction, temperature, and intensity to create a consistent lighting effect across the image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Read parameters
        light_direction = params.get("light_direction", (0.0, 0.0, 1.0))
        light_temperature = params.get("light_temperature", 5600.0)
        light_intensity = params.get("light_intensity", 1.03)

        # Normalize light direction
        direction = np.array(light_direction)
        direction = direction / np.linalg.norm(direction)

        # Create a lighting effect based on the parameters
        # Adjust brightness based on light intensity
        enhancer = ImageEnhance.Brightness(image)
        img_bright = enhancer.enhance(light_intensity)

        # Simulate color temperature adjustment
        img_array = np.array(img_bright)
        img_array = self.adjust_color_temperature(img_array, light_temperature)

        # Convert back to PIL Image
        result_image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        return result_image

    def adjust_color_temperature(self, img_array: np.ndarray, temperature: float) -> np.ndarray:
        # Scale the red and blue channels toward the target temperature.
        if temperature < 5000:
            # More blue
            img_array[..., 0] = img_array[..., 0] * (5000 / temperature)
        else:
            # More red
            img_array[..., 2] = img_array[..., 2] * (temperature / 5000)

        return img_array

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "light_direction": {
                "type": "array",
                "description": "Light direction vector [x, y, z].",
                "default": (0.0, 0.0, 1.0),
                "required": False
            },
            "light_temperature": {
                "type": "float",
                "min": 2000,
                "max": 10000,
                "description": "Light temperature in Kelvin.",
                "default": 5600.0,
                "required": False
            },
            "light_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Light intensity factor.",
                "default": 1.03,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: light_direction (vector), light_temperature (2000-10000 K), light_intensity (0.0-1.0). Use for harmonizing light in composite images."