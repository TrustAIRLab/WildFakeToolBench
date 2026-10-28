from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class BakedInImperfectionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="baked_in_imperfections",
            description="This tool adds realistic imperfections like film grain and lens flares to an image, enhancing its authenticity as a photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        imperfections = params.get("imperfections", "film_grain")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply imperfections based on the specified type
        if imperfections == "film_grain":
            image = self.add_film_grain(image)
        elif imperfections == "lens_flare":
            image = self.add_lens_flare(image)
        else:
            raise ValueError("Unsupported imperfections type.")

        return image

    def add_film_grain(self, image: Image.Image) -> Image.Image:
        # Create a film grain effect
        noise = np.random.normal(0, 25, (image.height, image.width, 3)).astype(np.uint8)
        noisy_image = np.array(image) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_lens_flare(self, image: Image.Image) -> Image.Image:
        # Create a simple lens flare effect
        flare = Image.new('RGBA', image.size, (255, 255, 255, 0))
        flare_radius = min(image.size) // 10
        for x in range(0, image.width, flare_radius):
            for y in range(0, image.height, flare_radius):
                flare.paste((255, 255, 255, 128), [x, y, x + flare_radius, y + flare_radius])
        flare = flare.filter(ImageFilter.GaussianBlur(10))
        return Image.alpha_composite(image.convert('RGBA'), flare).convert('RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "imperfections": {
                "type": "str",
                "description": "Type of imperfection to add (e.g., 'film_grain', 'lens_flare').",
                "default": "film_grain",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Choose 'film_grain' for a classic look or 'lens_flare' for a dramatic effect."