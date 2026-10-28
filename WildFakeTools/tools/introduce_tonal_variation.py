from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class IntroduceTonalVariationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="introduce_tonal_variation",
            description="This tool introduces subtle tonal variation and asymmetry to skin areas, making the image appear more lifelike and less uniform."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        variation_strength = params.get("variation_strength", 0.1)
        asymmetry_factor = params.get("asymmetry_factor", 0.05)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image).astype(np.float32)

        # Generate random noise for tonal variation
        noise = np.random.normal(0, variation_strength * 255, img_array.shape).astype(np.float32)

        # Apply asymmetry by adding noise with a random offset
        asymmetry_offset = np.random.uniform(-asymmetry_factor * 255, asymmetry_factor * 255, img_array.shape)
        img_array += noise + asymmetry_offset

        # Clip values to be in the valid range [0, 255] and convert back to uint8
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array, "RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "variation_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the tonal variation.",
                "default": 0.1,
                "required": False
            },
            "asymmetry_factor": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Factor to control the asymmetry in tonal variation.",
                "default": 0.05,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Variation strength: [0.0, 1.0], Asymmetry factor: [0.0, 1.0]. Use for enhancing skin tones."