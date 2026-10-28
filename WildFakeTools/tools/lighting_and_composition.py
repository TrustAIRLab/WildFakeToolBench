from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class LightingAndCompositionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="lighting_and_composition",
            description="Enhances the realism of an image by adjusting lighting and contrast, simulating low-key lighting and high contrast effects."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        
        # Apply low-key lighting effect by enhancing contrast and reducing brightness
        contrast_factor = params.get("contrast", 1.5)  # Default high contrast
        brightness_factor = params.get("brightness", 0.7)  # Default low brightness
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "contrast": {
                "type": "float",
                "min": 1.0,
                "max": 3.0,
                "description": "Adjusts the contrast of the image. Higher values increase contrast.",
                "default": 1.5,
                "required": False
            },
            "brightness": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Adjusts the brightness of the image. Lower values darken the image.",
                "default": 0.7,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust contrast (1.0-3.0) and brightness (0.0-1.0) for enhanced realism in images."