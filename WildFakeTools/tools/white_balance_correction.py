from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class WhiteBalanceCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="white_balance_correction",
            description="Adjusts the temperature and tint of the image to neutralize color casts, flattening the color palette to resemble a photograph taken with a real camera."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        temperature = params.get("temperature", 2.0)
        tint = params.get("tint", 0.0)

        # Convert image to RGB and then to numpy array
        image = image.convert("RGB")
        img_array = np.array(image, dtype=np.float32)

        # Normalize temperature and tint values
        temp_adjustment = (temperature / 100.0) * 255
        tint_adjustment = (tint / 100.0) * 255

        # Apply temperature adjustment
        img_array[..., 0] += temp_adjustment  # Red channel
        img_array[..., 1] += tint_adjustment  # Green channel
        img_array[..., 2] -= (temp_adjustment + tint_adjustment) / 2  # Blue channel

        # Clip values to be in the valid range [0, 255]
        img_array = np.clip(img_array, 0, 255)

        # Convert back to uint8 and create a new PIL image
        corrected_image = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        return corrected_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "temperature": {
                "type": "float",
                "min": -100.0,
                "max": 100.0,
                "description": "Adjusts the color temperature of the image.",
                "default": 0.0,
                "required": False
            },
            "tint": {
                "type": "float",
                "min": -100.0,
                "max": 100.0,
                "description": "Adjusts the tint of the image towards green or magenta.",
                "default": 0.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Temperature and tint values range from -100 to 100; use to neutralize color casts."