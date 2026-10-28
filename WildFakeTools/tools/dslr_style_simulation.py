from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class DslrStyleSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="dslr_style_simulation",
            description="Simulates characteristics like bokeh and film grain to make AI-generated images appear more like real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.5)
        effect_type = params.get("effect_type", "bokeh")

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply the selected effect
        if effect_type == 'bokeh':
            image = self.apply_bokeh(image, strength)
        elif effect_type == 'film grain':
            image = self.apply_film_grain(image, strength)

        return image

    def apply_bokeh(self, image: Image.Image, strength: float) -> Image.Image:
        # Create a blurred version of the image
        blur_radius = int(10 * strength)  # Scale blur radius based on strength
        blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Blend the original and blurred images
        return Image.blend(image, blurred_image, strength)

    def apply_film_grain(self, image: Image.Image, strength: float) -> Image.Image:
        # Create noise
        noise = np.random.normal(scale=255 * strength, size=(image.height, image.width, 3)).astype(np.uint8)
        noisy_image = np.array(image) + noise
        
        # Clip values to ensure they are within valid range
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the effect (0.0 = no effect, 1.0 = maximum effect).",
                "default": 0.5,
                "required": False
            },
            "effect_type": {
                "type": "str",
                "description": "Type of effect to apply ('bokeh' or 'film grain').",
                "default": "bokeh",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength: 0.0-1.0; Effect type: 'bokeh' or 'film grain'. Useful for enhancing AI-generated images."