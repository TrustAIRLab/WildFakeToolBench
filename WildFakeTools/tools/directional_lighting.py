from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class DirectionalLightingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="directional_lighting",
            description="Enhances the image with soft shadows and warm tones, mimicking natural lighting conditions found in real photography."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_type = params.get("lighting_type", "warm")
        intensity = params.get("intensity", "medium")

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Adjust brightness and contrast based on intensity
        if intensity == "low":
            factor = 0.8
        elif intensity == "high":
            factor = 1.2
        else:  # medium
            factor = 1.0

        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

        # Create a warm tone effect
        if lighting_type == "warm":
            warm_filter = np.array([1.1, 0.95, 0.85])  # Increase red, decrease green and blue
        else:  # default to neutral if not specified
            warm_filter = np.array([1.0, 1.0, 1.0])

        # Apply the warm filter
        img_array = img_array * warm_filter
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        result_image = Image.fromarray(img_array)

        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_type": {
                "type": "str",
                "description": "Type of lighting effect to apply (e.g., 'warm').",
                "default": "warm",
                "required": False
            },
            "intensity": {
                "type": "str",
                "description": "Intensity of the effect (low, medium, high).",
                "default": "medium",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity can be 'low', 'medium', or 'high'; use this tool to enhance realism in AI-generated images."