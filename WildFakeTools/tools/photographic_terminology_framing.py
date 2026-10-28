from typing import Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class PhotographicTerminologyFramingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="photographic_terminology_framing",
            description="This tool enhances an image by applying photography-related effects based on specific terminology. It adjusts sharpness and contrast to mimic real photographic qualities."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        prompt_text = params.get("prompt_text", "")
        term_importance = params.get("term_importance", 0.5)

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply sharpness based on term importance
        sharpness_factor = 1 + term_importance * 2  # Scale from 1 to 3
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)

        # Apply contrast based on term importance
        contrast_factor = 1 + term_importance * 1.5  # Scale from 1 to 2.5
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "prompt_text": {
                "type": "str",
                "description": "Photography terminology to guide the image enhancement.",
                "default": "",
                "required": False
            },
            "term_importance": {
                "type": "float",
                "min": 0,
                "max": 1,
                "description": "Importance of the photography terms, affecting sharpness and contrast.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust term_importance between 0 and 1 to control the effect strength; useful for enhancing AI-generated images."