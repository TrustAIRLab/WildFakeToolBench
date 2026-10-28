from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class IrisPoreDetailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="iris_pore_detailing",
            description="Enhances fine details in images by adding realistic skin pores and iris textures, making AI-generated images appear more lifelike."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.5)

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array
        img_array = np.array(image)

        # Apply detail enhancement using a Gaussian filter and blending
        detail = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=1))
        detail_array = np.array(detail)

        # Blend original and detail-enhanced images based on strength
        enhanced_array = (1 - strength) * img_array + strength * detail_array
        enhanced_image = Image.fromarray(np.clip(enhanced_array, 0, 255).astype(np.uint8))

        return enhanced_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the detail enhancement effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength ranges from 0.0 to 1.0; use higher values for more pronounced detail enhancement."