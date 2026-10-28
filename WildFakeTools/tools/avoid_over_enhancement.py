from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class AvoidOverEnhancementTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="avoid_over_enhancement",
            description="This tool applies a subtle blur and noise reduction to an image, making it appear more like a natural photograph and less like an overly enhanced AI-generated image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply a Gaussian blur to reduce sharpness
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Convert to numpy array for noise addition
        image_array = np.array(blurred_image)
        
        # Add slight noise to the image
        noise = np.random.normal(loc=0, scale=5, size=image_array.shape).astype(np.uint8)
        noisy_image_array = np.clip(image_array + noise, 0, 255)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(noisy_image_array)
        
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "prohibited_terms": {
                "type": "list",
                "description": "List of terms to avoid in prompts to prevent over-enhancement.",
                "default": [],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "This tool is most useful for images that may appear overly enhanced; apply subtle adjustments to enhance realism."