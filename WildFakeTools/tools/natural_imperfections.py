from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class NaturalImperfectionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="natural_imperfections",
            description="Adds natural imperfections like slight focus errors and lens aberrations to enhance realism."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        focus_error = params.get("focus_error", 0.1)
        aberration_strength = params.get("aberration_strength", 0.1)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_array = np.array(image)

        # Apply focus error (slight blurring)
        if focus_error > 0:
            blur_radius = focus_error * 5  # Scale the blur radius
            image_array = np.array(Image.fromarray(image_array).filter(ImageFilter.GaussianBlur(blur_radius)))

        # Apply lens aberration (color fringing)
        if aberration_strength > 0:
            red_channel = image_array[:, :, 0]
            green_channel = image_array[:, :, 1]
            blue_channel = image_array[:, :, 2]

            # Shift the red and blue channels slightly
            shift_amount = int(aberration_strength * 10)  # Scale the shift amount
            red_channel = np.roll(red_channel, shift_amount, axis=1)
            blue_channel = np.roll(blue_channel, -shift_amount, axis=1)

            # Combine channels back
            image_array = np.stack((red_channel, green_channel, blue_channel), axis=-1)

        return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "focus_error": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of focus error (0 to 1).",
                "default": 0.1,
                "required": False
            },
            "aberration_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of lens aberration (0 to 1).",
                "default": 0.1,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Focus error and aberration strength range from 0 to 1; useful for enhancing image realism."