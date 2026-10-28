from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class FaceSymmetryRepairTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="face_symmetry_repair",
            description="Enhances facial details and symmetry, making the image look more realistic and less AI-generated."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        restoration_strength = params.get("restoration_strength", 1.0)

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply a slight sharpening filter to enhance details
        sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(restoration_strength * 100), threshold=3))

        return sharpened_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "restoration_strength": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "description": "Strength of the face restoration effect.",
                "default": 1.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Restoration strength ranges from 0.0 to 2.0; higher values enhance facial details more."