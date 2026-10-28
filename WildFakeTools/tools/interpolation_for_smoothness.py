from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class InterpolationForSmoothnessTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="interpolation_for_smoothness",
            description="Applies a smoothing effect to an image, mimicking natural motion blur to distract from artificiality."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        frame_rate = params.get("frame_rate", 30)
        smoothness_level = params.get("smoothness_level", 0.5)

        # Convert image to numpy array and ensure it's in RGB format
        img_array = np.array(image.convert("RGB"))

        # Calculate the standard deviation for Gaussian filter based on smoothness level
        sigma = smoothness_level * 5  # Scale the smoothness level to a reasonable sigma value

        # Apply Gaussian filter to smooth the image
        smoothed_array = gaussian_filter(img_array, sigma=sigma)

        # Clip values to ensure they remain valid for image representation
        smoothed_array = np.clip(smoothed_array, 0, 255).astype(np.uint8)

        # Return as a new PIL Image
        return Image.fromarray(smoothed_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "frame_rate": {
                "type": "int",
                "min": 24,
                "max": 60,
                "description": "Frame rate for the motion interpolation effect.",
                "default": 30,
                "required": False
            },
            "smoothness_level": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Level of smoothness to apply, where 0.0 is no smoothness and 1.0 is maximum smoothness.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Smoothness level ranges from 0.0 to 1.0; useful for enhancing the fluidity of AI-generated images."