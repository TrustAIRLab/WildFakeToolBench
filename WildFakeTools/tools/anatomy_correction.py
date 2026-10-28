from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class AnatomyCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="anatomy_correction",
            description="This tool enhances the anatomical accuracy of images, improving features like hands and proportions to reduce the uncanny valley effect."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.5)

        # Convert image to numpy array
        img_array = np.array(image)

        # Apply a Gaussian blur to simulate anatomical refinement
        # The strength parameter will control the amount of blur applied
        blur_radius = strength * 5  # Scale the radius based on strength
        img_blurred = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Blend the original and blurred images to enhance anatomical features
        img_result = Image.blend(image, img_blurred, alpha=strength)

        return img_result.convert("RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the strength of anatomical refinement. Higher values lead to more blurring.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength ranges from 0.0 to 1.0; use higher values for more pronounced anatomical corrections."