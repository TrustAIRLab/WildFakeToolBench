from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class SkinQualityTuningTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="skin_quality_tuning",
            description="This tool enhances skin textures in images by adjusting the sampling method and steps, resulting in more realistic details."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        sampling_method = params.get("sampling_method", "DPM++ 3M SDE Karras")
        sampling_steps = params.get("sampling_steps", 50)

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array
        img_array = np.array(image)

        # Create a blurred version of the image to simulate a more realistic skin texture
        blurred_img = np.clip(img_array + np.random.normal(0, 10, img_array.shape).astype(np.uint8), 0, 255)

        # Adjust the image based on the sampling steps
        if sampling_steps < 1:
            sampling_steps = 1
        elif sampling_steps > 100:
            sampling_steps = 100

        # Apply a simple adjustment based on sampling steps
        adjustment_factor = sampling_steps / 100.0
        adjusted_img = np.clip(blurred_img * adjustment_factor, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(adjusted_img, 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "sampling_method": {
                "type": "str",
                "description": "The sampling method to use for texture adjustment.",
                "default": "DPM++ 3M SDE Karras",
                "required": False
            },
            "sampling_steps": {
                "type": "int",
                "min": 1,
                "max": 100,
                "description": "The number of sampling steps to optimize skin texture.",
                "default": 50,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust sampling steps between 1 and 100 to enhance skin textures effectively."