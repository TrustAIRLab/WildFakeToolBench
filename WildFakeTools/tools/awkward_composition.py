from typing import Dict, Any
from PIL import Image
import numpy as np
import random
from .base_tool import BaseTool


class AwkwardCompositionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="awkward_composition",
            description="Create images with off-center framing and messy angles to mimic accidental captures.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        framing_style = params.get("framing_style", "off-center")
        angle_variation = params.get("angle_variation", 0.1)

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Randomly rotate the image
        angle = random.uniform(-angle_variation * 45, angle_variation * 45)
        rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

        # Calculate the new size and crop to create an off-center effect
        width, height = rotated_image.size
        crop_x = int(random.uniform(0, width * 0.1))  # Crop up to 10% from the left
        crop_y = int(random.uniform(0, height * 0.1))  # Crop up to 10% from the top
        cropped_image = rotated_image.crop((crop_x, crop_y, width, height))

        return cropped_image.resize(image.size, Image.Resampling.BICUBIC)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "framing_style": {
                "type": "str",
                "description": "Style of framing, typically 'off-center'.",
                "default": "off-center",
                "required": False
            },
            "angle_variation": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Variation in angle for rotation, where 0 is no rotation and 1 is up to 45 degrees.",
                "default": 0.1,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Angle variation ranges from 0 to 1, influencing the degree of rotation; useful for creating spontaneous-looking images."