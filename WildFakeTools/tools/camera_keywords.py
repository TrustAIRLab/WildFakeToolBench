from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraKeywordsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_keywords",
            description="Enhances the quality of AI-generated images by simulating camera techniques such as depth of field and sharpening, making them resemble real photographs more closely.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        camera_keywords = params.get("camera_keywords", "shallow depth of field, sharpen")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        np_image = np.array(image)

        # Apply effects based on keywords
        if "shallow depth of field" in camera_keywords:
            np_image = self.apply_depth_of_field(np_image)
        if "sharpen" in camera_keywords:
            np_image = self.apply_sharpen(np_image)

        # Ensure the output is in uint8 format
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(np_image)

    def apply_depth_of_field(self, np_image: np.ndarray) -> np.ndarray:
        # Simple Gaussian blur to simulate shallow depth of field
        blurred_image = Image.fromarray(np_image).filter(ImageFilter.GaussianBlur(radius=5))
        return np.array(blurred_image)

    def apply_sharpen(self, np_image: np.ndarray) -> np.ndarray:
        # Simple sharpening filter
        sharpened_image = Image.fromarray(np_image).filter(ImageFilter.SHARPEN)
        return np.array(sharpened_image)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera_keywords": {
                "type": "str",
                "description": "Comma-separated camera-related keywords to enhance image quality.",
                "default": "shallow depth of field, sharpen",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use keywords like 'shallow depth of field' and 'sharpen' to enhance image quality."