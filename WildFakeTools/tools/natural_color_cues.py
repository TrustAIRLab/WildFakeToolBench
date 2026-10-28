from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class NaturalColorCuesTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="natural_color_cues",
            description="Adjusts color balance and saturation to create a more natural and muted color palette, mimicking casual phone photography."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        color_balance = params.get("color_balance", 0.0)
        saturation = params.get("saturation", 0.5)

        # Convert image to RGB and then to numpy array
        image = image.convert("RGB")
        img_array = np.array(image).astype(np.float32) / 255.0

        # Adjust saturation
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        img_array = img_array * (1 - saturation) + gray[..., np.newaxis] * saturation

        # Adjust color balance
        warm_filter = np.array([1 + color_balance, 1 - color_balance, 1 - color_balance])
        img_array = img_array * warm_filter

        # Clip values to be in the valid range [0, 1]
        img_array = np.clip(img_array, 0, 1)

        # Convert back to PIL Image
        result_image = Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "color_balance": {
                "type": "float",
                "min": -1.0,
                "max": 1.0,
                "description": "Adjusts the warmth of the image, where negative values add cool tones and positive values add warm tones.",
                "default": 0.0,
                "required": False
            },
            "saturation": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the saturation level of the image, where 0.0 is grayscale and 1.0 is fully saturated.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust color balance (-1.0 to 1.0) and saturation (0.0 to 1.0) for more natural-looking images."