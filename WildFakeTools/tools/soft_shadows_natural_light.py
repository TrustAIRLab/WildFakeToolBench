from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SoftShadowsNaturalLightTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="soft_shadows_natural_light",
            description="Applies soft shadows and natural lighting effects to enhance the dimensionality and realism of an image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        shadow_type = params.get("shadow_type", "soft")
        light_conditions = params.get("light_conditions", "natural")

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_np = np.array(image)

        # Create a shadow effect
        if shadow_type == "soft":
            shadow = Image.fromarray(image_np).filter(ImageFilter.GaussianBlur(radius=10))
        else:  # hard shadow
            shadow = Image.fromarray(image_np).filter(ImageFilter.GaussianBlur(radius=0))

        # Create a light effect
        if light_conditions == "natural":
            light_intensity = 1.2
        else:  # artificial
            light_intensity = 0.8

        # Blend the original image with the shadow and adjust brightness
        shadow_np = np.array(shadow)
        blended_image_np = np.clip(image_np * light_intensity + shadow_np * (1 - light_intensity), 0, 255).astype(np.uint8)

        return Image.fromarray(blended_image_np, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "shadow_type": {
                "type": "str",
                "description": "Type of shadow to apply: 'soft' for soft shadows, 'hard' for hard shadows.",
                "default": "soft",
                "required": False
            },
            "light_conditions": {
                "type": "str",
                "description": "Lighting conditions: 'natural' for natural light, 'artificial' for artificial light.",
                "default": "natural",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: shadow_type ('soft' or 'hard'), light_conditions ('natural' or 'artificial'). Use for enhancing realism."