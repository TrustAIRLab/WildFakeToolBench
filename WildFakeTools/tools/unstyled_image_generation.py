from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class UnstyledImageGenerationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="unstyled_image_generation",
            description="This tool applies a series of filters to create an unrefined, natural look to images, reducing the polished appearance typical of AI-generated photos."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        unstyled = params.get("unstyled", True)

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if unstyled:
            # Apply a Gaussian blur to soften sharp features
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            # Add noise to the image
            noise = np.random.normal(loc=0, scale=25, size=(image.height, image.width, 3)) * 0.16
            noisy_image = np.array(image, dtype=np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_image)

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "unstyled": {
                "type": "bool",
                "description": "Whether to apply unstyled effects to the image.",
                "default": True,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Set 'unstyled' to True to reduce polished features and enhance naturalness in the image."