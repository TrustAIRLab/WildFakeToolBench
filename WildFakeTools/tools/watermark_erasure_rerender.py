from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class WatermarkErasureRerenderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="watermark_erasure_rerender",
            description="This tool enhances an AI-generated image by applying a blur effect and adjusting colors to make it resemble a real photograph, effectively discarding any watermarks."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Create a blurred version of the image to simulate a reference
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Convert images to numpy arrays for manipulation
        original_array = np.array(image)
        blurred_array = np.array(blurred_image)

        # Adjust the original image's colors towards the blurred image's colors
        adjusted_array = np.clip(original_array * 0.7 + blurred_array * 0.3, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(adjusted_array, mode='RGB')
        
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "reference_image": {
                "type": "str",
                "description": "Path to the reference image (not used in this implementation).",
                "default": "",
                "required": False
            },
            "rendering_style": {
                "type": "str",
                "description": "Style of rendering (e.g., 'realistic', 'artistic').",
                "default": "realistic",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters can be adjusted for different rendering styles; default is 'realistic'."