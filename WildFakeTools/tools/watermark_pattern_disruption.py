from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class WatermarkPatternDisruptionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="watermark_pattern_disruption",
            description="This tool applies noise and blurring to an image to disrupt patterns and create a more natural appearance, helping to remove artifacts like watermarks."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        noise_strength = params.get("noise_strength", 0.1)
        iterations = params.get("iterations", 5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Generate noise
        noise = np.random.normal(0, noise_strength * 255, img_array.shape).astype(np.uint8)

        # Add noise to image
        noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        noisy_image_pil = Image.fromarray(noisy_image)

        # Apply Gaussian blur to smooth out the noise
        for _ in range(iterations):
            noisy_image_pil = noisy_image_pil.filter(ImageFilter.GaussianBlur(radius=2))

        return noisy_image_pil

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "noise_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the noise to be added to the image.",
                "default": 0.1,
                "required": False
            },
            "iterations": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Number of iterations for applying Gaussian blur.",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Noise strength: 0.0 to 1.0; iterations: 1 to 10. Use this tool to reduce artifacts and create a more natural image appearance."