from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class SpecificLightingDescriptionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="specific_lighting_descriptions",
            description="Enhances the realism of an image by adjusting brightness and contrast based on specified lighting conditions.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_description = params.get("lighting_description", "bright, low contrast")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Adjust image based on lighting description
        if "bright" in lighting_description:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)  # Increase brightness
        elif "dim" in lighting_description:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)  # Decrease brightness

        if "high contrast" in lighting_description:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
        elif "low contrast" in lighting_description:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(0.7)  # Decrease contrast

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_description": {
                "type": "str",
                "description": "A description of the desired lighting conditions (e.g., 'bright', 'dim', 'high contrast').",
                "default": "bright, low contrast",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Specify lighting conditions like 'bright', 'dim', or 'high contrast' to enhance realism."