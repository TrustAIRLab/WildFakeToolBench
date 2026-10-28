from typing import Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class QualityDescriptorBoostTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="quality_descriptor_boost",
            description="This tool enhances the realism of AI-generated images by applying filters and adjustments that mimic high-quality photographic attributes."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        quality_descriptors = params.get("quality_descriptors", "sharp, contrast")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancements based on quality descriptors
        if "sharp" in quality_descriptors:
            image = image.filter(ImageFilter.SHARPEN)
        if "smooth" in quality_descriptors:
            image = image.filter(ImageFilter.SMOOTH)
        if "contrast" in quality_descriptors:
            image = ImageEnhance.Contrast(image).enhance(1.5)
        if "brightness" in quality_descriptors:
            image = ImageEnhance.Brightness(image).enhance(1.2)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "quality_descriptors": {
                "type": "str",
                "description": "A string of quality-related descriptors (e.g., 'sharp, smooth, contrast').",
                "default": "sharp, contrast",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use descriptors like 'sharp', 'smooth', 'contrast' to enhance image realism."