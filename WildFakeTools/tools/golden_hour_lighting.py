from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class GoldenHourLightingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="golden_hour_lighting",
            description="Enhances images to simulate the warm, soft lighting of the golden hour, making them appear more like professional photographs.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        lighting_condition = params.get("lighting_condition", "golden_hour")

        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array for manipulation
        img_array = np.array(image)

        # Simulate golden hour lighting by applying a warm color filter
        if lighting_condition == "golden_hour":
            # Increase red and green channels to create a warm effect
            img_array = img_array.astype(np.float32)  # Convert to float for manipulation
            img_array[..., 0] *= 1.2  # Increase red channel
            img_array[..., 1] *= 1.1  # Increase green channel
            img_array = np.clip(img_array, 0, 255)  # Ensure values are within valid range

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_array.astype(np.uint8), 'RGB')
        return enhanced_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_condition": {
                "type": "str",
                "description": "The type of lighting condition to simulate.",
                "default": "golden_hour",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use with 'golden_hour' for warm lighting effects; adjust for different lighting conditions."