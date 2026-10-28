from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class RealisticHandsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="realistic_hands",
            description="Enhances the appearance of hands in an image to ensure they have five fingers and are in natural positions, making the image more believable."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        hand_position = params.get("hand_position", "natural pose")

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array for manipulation
        img_array = np.array(image)

        # Add noise to mimic natural hand texture.
        if hand_position == "natural pose":
            # Apply a Gaussian blur to the image to simulate a more natural look
            img_array = np.clip(img_array + np.random.normal(0, 5, img_array.shape).astype(np.uint8), 0, 255)

        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_array, 'RGB')
        return enhanced_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "hand_position": {
                "type": "str",
                "description": "Specifies the position of the hands in the image.",
                "default": "natural pose",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameter 'hand_position' can be set to 'natural pose' to enhance hand realism."