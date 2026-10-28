from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class HighResolutionGenerationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="high_resolution_generation",
            description="Enhances images by generating them at higher resolutions, adding intricate details and noise patterns that resemble real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        resolution = params.get("resolution", 2048)
        aspect_ratio = params.get("aspect_ratio", 1.0)

        # Calculate new dimensions
        width = resolution
        height = int(resolution / aspect_ratio)

        # Resize the image to the new dimensions
        image = image.convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)

        # Apply a slight blur to mimic natural detail
        image = image.filter(ImageFilter.GaussianBlur(radius=1))

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "resolution": {
                "type": "int",
                "min": 2048,
                "max": 4096,
                "description": "The target resolution for the generated image.",
                "default": 2048,
                "required": False
            },
            "aspect_ratio": {
                "type": "float",
                "min": 0.5,
                "max": 2.0,
                "description": "The aspect ratio of the image (width/height).",
                "default": 1.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Resolution should be between 2048 and 4096; aspect ratio should be between 0.5 and 2.0 for best results."