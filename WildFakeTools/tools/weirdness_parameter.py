from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class WeirdnessParameterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="weirdness_parameter",
            description="Adjusts the 'weirdness' parameter to introduce randomness and diversity in images, making them more varied and lifelike."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        weirdness = params.get("weirdness", 0.5)

        # Convert image to numpy array
        img_array = np.array(image)

        # Introduce randomness based on weirdness parameter
        noise = np.random.normal(loc=0, scale=weirdness * 25, size=img_array.shape).astype(np.uint8)
        weird_image_array = np.clip(img_array + noise, 0, 255)

        # Convert back to PIL Image
        weird_image = Image.fromarray(weird_image_array, 'RGB')
        return weird_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "weirdness": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the level of randomness introduced into the image.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Weirdness parameter ranges from 0.0 to 1.0; higher values introduce more randomness for unique images."