from typing import Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class ClarityEnhancerPolishTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="clarity_enhancer_polish",
            description="Enhances the clarity of the image by sharpening and improving micro-contrast, making it crisper and more defined."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        sharpening_strength = params.get("sharpening_strength", 0.5)
        contrast_level = params.get("contrast_level", 0.5)

        # Convert image to RGB if needed
        image = image.convert("RGB")

        # Apply sharpening
        sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpening_strength * 100), threshold=3))

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(sharpened_image)
        enhanced_image = enhancer.enhance(1 + contrast_level)

        return enhanced_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "sharpening_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the sharpening effect.",
                "default": 0.5,
                "required": False
            },
            "contrast_level": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Level of contrast enhancement.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters range from 0.0 to 1.0 for both sharpening and contrast; useful for enhancing image clarity."