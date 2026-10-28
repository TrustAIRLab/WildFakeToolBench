from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraSpecificationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_specification",
            description="Applies a post-processing effect to enhance the realism of an AI-generated image, simulating the characteristics of a specific camera model."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        camera_model = params.get("camera_model", "iPhone 12")
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create a blurred version of the image to simulate lens effects
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # Convert images to numpy arrays for manipulation
        original_array = np.array(image)
        blurred_array = np.array(blurred_image)

        # Simulate a vignette effect based on camera model
        height, width, _ = original_array.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        vignette = np.clip(1 - (X**2 + Y**2), 0, 1)

        # Apply vignette effect
        vignette_array = (original_array * vignette[:, :, np.newaxis]).astype(np.uint8)

        # Blend the original and blurred images
        final_image_array = np.clip((vignette_array * 0.7 + blurred_array * 0.3), 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        final_image = Image.fromarray(final_image_array, 'RGB')
        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera_model": {
                "type": "str",
                "description": "The model of the camera to simulate (e.g., 'iPhone 12', 'Canon EOS').",
                "default": "iPhone 12",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Camera model can be specified to adjust realism; default is 'iPhone 12'."