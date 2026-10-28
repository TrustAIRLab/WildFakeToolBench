from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from .base_tool import BaseTool


class KeywordPrimingForHandsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="keyword_priming_for_hands",
            description="Enhances the details of hands in an image by applying sharpening and contrast adjustments."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        hand_keywords = params.get("hand_keywords", "beautiful hands, detailed fingers")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast slightly
        
        # Apply a sharpening filter
        sharpener = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        image = image.filter(sharpener)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "hand_keywords": {
                "type": "str",
                "description": "Keywords to prime the image enhancement for hands.",
                "default": "beautiful hands, detailed fingers",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use keywords to enhance hand details; default is 'beautiful hands, detailed fingers'."