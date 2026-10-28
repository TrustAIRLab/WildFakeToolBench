from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class HarmonizeLightingColorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="harmonize_lighting_color",
            description="Adjusts the lighting direction, temperature, and intensity to create a consistent visual environment across the image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters
        light_direction = np.array(params.get("light_direction", [0.0, 0.0, 1.0]))
        light_temperature = params.get("light_temperature", 6500.0)
        light_intensity = params.get("light_intensity", 0.5)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_array = np.array(image)

        # Normalize light direction
        light_direction = light_direction / np.linalg.norm(light_direction)

        # Create a lighting effect based on the parameters
        # Simulate the effect of light temperature
        temperature_scale = (light_temperature - 2000) / (10000 - 2000)
        r_factor = 1 + (temperature_scale * 0.5)  # Red channel adjustment
        b_factor = 1 - (temperature_scale * 0.5)  # Blue channel adjustment

        # Apply lighting intensity and color adjustment
        adjusted_image = image_array * light_intensity
        adjusted_image[..., 0] *= r_factor  # Adjust red channel
        adjusted_image[..., 2] *= b_factor  # Adjust blue channel

        # Clip values to ensure they remain valid
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

        return Image.fromarray(adjusted_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "light_direction": {
                "type": "array",
                "description": "A 3D vector [x, y, z] representing the direction of the light source.",
                "default": [0.0, 0.0, 1.0],
                "required": False
            },
            "light_temperature": {
                "type": "float",
                "min": 2000,
                "max": 10000,
                "description": "The color temperature of the light source in Kelvin.",
                "default": 6500.0,
                "required": False
            },
            "light_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "The intensity of the light effect applied to the image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: light_direction (3D vector), light_temperature (2000-10000), light_intensity (0.0-1.0). Use to harmonize lighting in composite images."