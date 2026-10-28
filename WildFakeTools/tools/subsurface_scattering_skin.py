from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SubsurfaceScatteringSkinTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="subsurface_scattering_skin",
            description="This tool introduces natural imperfections and subsurface scattering to mimic real skin texture, enhancing surface realism and helping evade detection."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create a noise texture to simulate skin imperfections
        noise = np.random.normal(loc=0, scale=strength * 20, size=img_array.shape).astype(np.uint8)
        
        # Add noise to the image
        textured_image_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        textured_image = Image.fromarray(textured_image_array)

        # Apply a slight blur to simulate subsurface scattering
        textured_image = textured_image.filter(ImageFilter.GaussianBlur(radius=strength * 2))

        return textured_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the intensity of the skin texture effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength ranges from 0.0 (no effect) to 1.0 (maximum texture); useful for enhancing realism in AI-generated images."