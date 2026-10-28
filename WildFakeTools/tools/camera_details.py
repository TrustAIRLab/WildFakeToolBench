from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_details",
            description="Enhances the photographic quality of images by simulating camera characteristics such as lens type and depth of field."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        lens_type = params.get("lens_type", "wide")
        depth_of_field = params.get("depth_of_field", "shallow")

        # Apply effects based on lens type
        if lens_type == "wide":
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        elif lens_type == "telephoto":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Simulate depth of field effect
        if depth_of_field == "shallow":
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        elif depth_of_field == "deep":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lens_type": {
                "type": "str",
                "description": "Type of lens to simulate (e.g., 'standard', 'wide', 'telephoto').",
                "default": "wide",
                "required": False
            },
            "depth_of_field": {
                "type": "str",
                "description": "Depth of field effect (e.g., 'shallow', 'medium', 'deep').",
                "default": "shallow",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Lens types: 'standard', 'wide', 'telephoto'; Depth of field: 'shallow', 'medium', 'deep'."