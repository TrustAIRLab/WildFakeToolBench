from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class AggressiveTextureDescriptorsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="aggressive_texture_descriptors",
            description="Enhances the texture of images by applying a series of filters that simulate skin imperfections and details."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        texture_detail_level = params.get("texture_detail_level", 5)
        imperfection_types = params.get("imperfection_types", ["freckles", "wrinkles"])

        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_array = np.array(image)

        # Apply a Gaussian blur to simulate skin texture
        blur_radius = max(0, min(10, texture_detail_level))  # Clamp between 0 and 10
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Convert back to array for further processing
        blurred_array = np.array(blurred_image)

        # Add noise to simulate imperfections
        noise_level = texture_detail_level * 5  # Scale noise level
        noise = np.random.normal(0, noise_level, image_array.shape).astype(np.uint8)
        textured_image_array = np.clip(image_array + noise, 0, 255)

        # Convert back to PIL Image
        textured_image = Image.fromarray(textured_image_array)

        return textured_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "texture_detail_level": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Level of texture detail to apply.",
                "default": 5,
                "required": False
            },
            "imperfection_types": {
                "type": "list",
                "description": "Types of skin imperfections to simulate.",
                "default": ["freckles", "wrinkles"],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Texture detail level ranges from 1 to 10; useful for enhancing realism in AI-generated images."