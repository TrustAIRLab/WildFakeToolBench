from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class DeviceSpecificityTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="device_specificity",
            description="Applies specific filters and styles to mimic the characteristics of images taken with a specified phone model."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        phone_model = params.get("phone_model", "default")

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply different filters based on the phone model
        if phone_model.lower() == "iphone":
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        elif phone_model.lower() == "samsung":
            image = image.filter(ImageFilter.SHARPEN)
        elif phone_model.lower() == "google pixel":
            image = image.filter(ImageFilter.CONTOUR)
        else:
            image = image.filter(ImageFilter.DETAIL)

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "phone_model": {
                "type": "str",
                "description": "The model of the phone to mimic (e.g., 'iPhone', 'Samsung', 'Google Pixel').",
                "default": "default",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Specify a phone model to apply corresponding filters; options include 'iPhone', 'Samsung', 'Google Pixel'."