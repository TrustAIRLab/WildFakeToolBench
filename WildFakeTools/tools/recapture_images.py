from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class RecaptureImagesTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="recapture_images",
            description="This tool simulates the effect of capturing an AI-generated image displayed on a screen or printed out, creating a 'camera original' image that appears more authentic."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply a slight blur to simulate camera lens effects
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Add noise to simulate the imperfections of a captured image
        noise = np.random.normal(0, 10, (blurred_image.height, blurred_image.width, 3)).astype(np.uint8)
        noisy_image = np.array(blurred_image) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        final_image = Image.fromarray(noisy_image)
        
        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "capture_device": {
                "type": "str",
                "description": "Type or model of the camera used for capturing.",
                "default": "generic_camera",
                "required": False
            },
            "display_type": {
                "type": "str",
                "description": "Type of display used (screen or print).",
                "default": "screen",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: capture_device (str), display_type (str). Use this tool to create a more authentic appearance for AI-generated images."