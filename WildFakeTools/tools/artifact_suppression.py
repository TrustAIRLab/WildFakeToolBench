from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ArtifactSuppressionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="artifact_suppression",
            description="This tool enhances AI-generated images by applying filters and adjustments to reduce undesired artistic qualities, making the image appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        negative_keywords = params.get("negative_keywords", "paintings, low quality")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply filters and adjustments based on negative keywords
        if "paintings" in negative_keywords:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Blur to reduce sharp artistic features
        if "low quality" in negative_keywords:
            image = image.filter(ImageFilter.SHARPEN)  # Sharpen to enhance details
            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))  # Further enhance details
        
        # Convert back to RGB just in case
        return image.convert('RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "negative_keywords": {
                "type": "str",
                "description": "Keywords describing undesirable qualities to avoid in the image.",
                "default": "paintings, low quality",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use negative keywords to refine image quality; e.g., 'paintings', 'low quality'."