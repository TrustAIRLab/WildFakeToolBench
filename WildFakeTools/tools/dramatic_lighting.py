from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class DramaticLightingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="dramatic_lighting",
            description="Enhances the image by simulating specific lighting conditions like 'golden hour' or 'soft sunlight' to increase realism."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_condition = params.get("lighting_condition", "golden hour")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply lighting effects based on the specified condition
        if lighting_condition == "golden hour":
            # Simulate warm lighting by adjusting color balance
            r, g, b = image.split()
            r = r.point(lambda i: i * 1.2)  # Increase red channel
            g = g.point(lambda i: i * 0.9)   # Decrease green channel
            image = Image.merge('RGB', (r, g, b))
        
        elif lighting_condition == "soft sunlight":
            # Simulate soft sunlight by enhancing brightness and contrast
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)  # Slightly increase brightness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)   # Slightly increase contrast
        
        # Return the modified image
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_condition": {
                "type": "str",
                "description": "The type of lighting condition to simulate.",
                "default": "golden hour",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Choose 'golden hour' for warm tones or 'soft sunlight' for a gentle brightness effect."