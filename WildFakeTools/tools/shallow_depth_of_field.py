from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ShallowDepthOfFieldTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="shallow_depth_of_field",
            description="Simulates a shallow depth of field effect by blurring the background of the image while keeping the subject in focus, mimicking the effect of smartphone camera lenses."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters
        aperture = params.get("aperture", 1.8)
        blur_strength = params.get("blur_strength", 0.5)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a mask for the depth of field effect
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.float32)

        # Simulate a circular gradient mask for depth of field
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y) * 0.8
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance < max_radius:
                    mask[y, x] = 1.0 - (distance / max_radius)

        # Apply blur based on the mask
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=aperture))
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Blend the original and blurred images based on the mask and blur strength
        final_image = Image.composite(image, blurred_image, mask_image)
        
        # Adjust the final image based on blur strength
        final_image = Image.blend(image, final_image, blur_strength)

        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "aperture": {
                "type": "float",
                "min": 1.2,
                "max": 2.8,
                "description": "Aperture value simulating lens opening (1.2 = wide, 2.8 = narrow).",
                "default": 1.8,
                "required": False
            },
            "blur_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the blur effect applied to the background.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Aperture: 1.2-2.8; Blur strength: 0.0-1.0. Use to create a natural background blur effect."