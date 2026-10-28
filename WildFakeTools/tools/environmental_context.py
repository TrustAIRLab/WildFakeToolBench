from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from .base_tool import BaseTool


class EnvironmentalContextTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="environmental_context",
            description="Enhances an image by simulating realistic lighting conditions and environmental context, making it appear more like a natural photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters with defaults
        lighting_condition = params.get("lighting_condition", "golden hour")
        environment = params.get("environment", "coffee shop")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Simulate lighting effects
        if lighting_condition == "golden hour":
            image = self.apply_golden_hour_effect(image)
        
        # Simulate environmental context
        if environment == "coffee shop":
            image = self.add_coffee_shop_effect(image)
        
        return image

    def apply_golden_hour_effect(self, image: Image.Image) -> Image.Image:
        # Apply a warm color filter to simulate golden hour lighting
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)  # Increase color saturation
        image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Slight blur for softness
        return image

    def add_coffee_shop_effect(self, image: Image.Image) -> Image.Image:
        # Add a slight vignette effect to simulate a cozy indoor environment
        np_image = np.array(image)
        rows, cols, _ = np_image.shape
        # Create a vignette mask
        X_resultant_kernel = np.tile(np.linspace(0, 1, cols), (rows, 1))
        Y_resultant_kernel = np.tile(np.linspace(0, 1, rows), (cols, 1)).T
        kernel = np.sqrt(X_resultant_kernel**2 + Y_resultant_kernel**2)
        kernel = np.clip(kernel, 0, 1)
        # Apply the vignette effect
        np_image = np_image * (1 - 0.5 * kernel[:, :, np.newaxis])
        return Image.fromarray(np.uint8(np_image))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_condition": {
                "type": "str",
                "description": "The lighting condition to simulate.",
                "default": "golden hour",
                "required": False
            },
            "environment": {
                "type": "str",
                "description": "The environmental context to simulate.",
                "default": "coffee shop",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: lighting_condition (e.g., 'natural light', 'golden hour'), environment (e.g., 'park', 'coffee shop'). Use to enhance realism."