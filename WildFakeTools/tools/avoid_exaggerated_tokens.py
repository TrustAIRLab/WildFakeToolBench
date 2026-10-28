from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class AvoidExaggeratedTokensTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="avoid_exaggerated_tokens",
            description="This tool reduces the sharpness and polish of an image, making it appear more realistic by applying a Gaussian blur.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        # Convert image to RGB if needed
        image = image.convert("RGB")
        # Convert image to numpy array
        image_array = np.array(image)
        # Apply Gaussian blur to reduce sharpness
        blurred_image_array = gaussian_filter(image_array, sigma=1)
        # Convert back to PIL Image
        return Image.fromarray(np.uint8(blurred_image_array))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "prohibited_tokens": {
                "type": "list",
                "description": "List of prohibited tokens to avoid in prompts.",
                "default": [],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use this tool to soften images by applying a Gaussian blur, making them appear more realistic. Adjust the parameters as needed."