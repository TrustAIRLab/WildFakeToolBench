from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class NoiseDistortionAdditionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="noise_distortion_addition",
            description="Adds noise and distortion to images to disrupt detection patterns, masking statistical signatures typically recognized by AI detectors."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        noise_level = params.get("noise_level", 0.1)
        distortion_type = params.get("distortion_type", "none")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_array = np.array(image)

        # Add noise
        noise = np.random.normal(0, noise_level * 255, image_array.shape).astype(np.uint8)
        noisy_image = np.clip(image_array + noise, 0, 255)

        # Apply distortion
        if distortion_type == "blur":
            from scipy.ndimage import gaussian_filter
            noisy_image = gaussian_filter(noisy_image, sigma=1)

        elif distortion_type == "pixelate":
            pixelation_size = 8
            small_image = Image.fromarray(noisy_image.astype(np.uint8)).resize(
                (max(1, image.width // pixelation_size), max(1, image.height // pixelation_size)),
                Image.Resampling.LANCZOS
            )
            noisy_image = np.array(small_image.resize(image.size, Image.Resampling.LANCZOS))

        return Image.fromarray(noisy_image.astype(np.uint8))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "noise_level": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Level of noise to add to the image.",
                "default": 0.1,
                "required": False
            },
            "distortion_type": {
                "type": "str",
                "description": "Type of distortion to apply (none, blur, pixelate).",
                "default": "none",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Noise level: 0.0 to 1.0; Distortion type: none, blur, pixelate. Useful for masking AI detection patterns."