from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class StyleRawUsageTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="style_raw_usage",
            description="This tool reduces beautification effects in an image, allowing it to adhere more closely to realistic light and shadow physics."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        style = params.get("style", "soft")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply a filter based on style
        if style == "normal":
            return image  # No changes for normal style
        elif style == "soft":
            return image.filter(ImageFilter.SMOOTH)  # Softening the image slightly
        elif style == "sharp":
            return image.filter(ImageFilter.SHARPEN)  # Sharpening the image slightly
        # Unrecognised styles use the default.
        return image.filter(ImageFilter.SMOOTH)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "style": {
                "type": "str",
                "description": "The style to apply to the image. Options are 'normal', 'soft', or 'sharp'.",
                "default": "soft",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use styles 'normal', 'soft', or 'sharp' to adjust the realism of the image."