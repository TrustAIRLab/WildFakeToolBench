from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class LightingKeywordsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="lighting_keywords",
            description="Enhances the realism of an image by simulating various lighting conditions. Adjusts brightness, contrast, and color balance based on specified lighting keywords."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_keywords = params.get("lighting_keywords", "warm, high contrast").lower()

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Adjust image based on lighting keywords
        if "bright" in lighting_keywords:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)  # Increase brightness
        elif "dim" in lighting_keywords:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)  # Decrease brightness

        if "high contrast" in lighting_keywords:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
        elif "low contrast" in lighting_keywords:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(0.7)  # Decrease contrast

        if "warm" in lighting_keywords:
            r, g, b = image.split()
            r = r.point(lambda i: i * 1.1)  # Increase red channel
            image = Image.merge("RGB", (r, g, b))
        elif "cool" in lighting_keywords:
            r, g, b = image.split()
            b = b.point(lambda i: i * 1.1)  # Increase blue channel
            image = Image.merge("RGB", (r, g, b))

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_keywords": {
                "type": "str",
                "description": "Keywords describing the desired lighting effect (e.g., 'bright', 'dim', 'high contrast', 'warm').",
                "default": "warm, high contrast",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Specify lighting keywords to adjust brightness, contrast, and color balance for enhanced realism."