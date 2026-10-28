from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class StylizationExclusionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="stylization_exclusion",
            description="This tool applies a series of filters to an image to reduce non-photorealistic elements, simulating the exclusion of certain terms. It enhances realism by blurring and adjusting colors to align more closely with natural aesthetics."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        excluded_terms = params.get("excluded_terms", [])
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        
        # Apply a Gaussian blur to reduce sharpness and enhance realism
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Convert image to numpy array for further processing
        img_array = np.array(image)
        
        # Adjust the color balance slightly to simulate a more natural look
        img_array = img_array.astype(np.float32)
        img_array[..., 0] = np.clip(img_array[..., 0] * 1.1, 0, 255)  # Increase red channel
        img_array[..., 1] = np.clip(img_array[..., 1] * 0.95, 0, 255)  # Decrease green channel
        img_array[..., 2] = np.clip(img_array[..., 2] * 0.9, 0, 255)   # Decrease blue channel
        
        # Convert back to uint8 and create a new PIL image
        img_array = img_array.astype(np.uint8)
        processed_image = Image.fromarray(img_array)
        
        return processed_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "excluded_terms": {
                "type": "list",
                "description": "List of terms to exclude for enhancing realism.",
                "default": [],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use with a list of excluded terms to refine the image's realism. Adjusts colors and applies blurring for a more natural appearance."