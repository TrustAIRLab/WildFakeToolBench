from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraAndFilmKeywordsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_and_film_keywords",
            description="Enhances an image to mimic characteristics of real photographs by applying effects associated with specific camera and film types.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        keywords = params.get("keywords", "RAW, DSLR")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply enhancements based on keywords
        if "RAW" in keywords:
            image = self.apply_raw_effects(image)
        if "DSLR" in keywords:
            image = self.apply_dslr_effects(image)

        return image

    def apply_raw_effects(self, image: Image.Image) -> Image.Image:
        # Simulate the effect of a RAW photo by enhancing contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness
        return image

    def apply_dslr_effects(self, image: Image.Image) -> Image.Image:
        # Simulate a DSLR effect by adding a slight blur and adjusting brightness
        image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Slight blur
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # Slightly increase brightness
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "keywords": {
                "type": "str",
                "description": "Comma-separated camera and film keywords to simulate effects.",
                "default": "RAW, DSLR",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use keywords like 'RAW' or 'DSLR' to enhance the image; effects are applied based on the specified keywords."