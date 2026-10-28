from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class AspectRatioCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="aspect_ratio_correction",
            description="This tool enforces a 1:1 aspect ratio on images to enhance anatomical accuracy, preventing distortion in body proportions."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        # Convert image to RGB if needed
        image = image.convert("RGB")
        width, height = image.size
        
        # Calculate the size for the new square image
        new_size = max(width, height)
        
        # Create a new square image with a white background
        new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        
        # Paste the original image onto the center of the new image
        new_image.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
        
        return new_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "aspect_ratio": {
                "type": "str",
                "description": "Aspect ratio to enforce, currently only '1:1' is supported.",
                "default": "1:1",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "This tool enforces a 1:1 aspect ratio; it is most useful for images where anatomical accuracy is critical."