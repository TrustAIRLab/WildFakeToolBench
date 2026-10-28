from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraGearRealismTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_gear_realism",
            description="This tool enhances the realism of AI-generated images by simulating the characteristics of specific camera models and lens details, making the output appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        camera_model = params.get("camera_model", "Canon EOS 5D Mark IV")
        lens_details = params.get("lens_details", "EF 24-70mm f/2.8L II USM")

        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Simulate a depth of field effect by blurring the background
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))

        # Create a mask to simulate focus on the center
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        center_x, center_y = image.width // 2, image.height // 2
        radius = min(center_x, center_y) // 2
        y, x = np.ogrid[:image.height, :image.width]
        mask_area = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[mask_area] = 255

        # Blend the original image with the blurred image using the mask
        final_image = Image.composite(image, blurred_image, Image.fromarray(mask))

        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera_model": {
                "type": "str",
                "description": "The camera model to simulate.",
                "default": "Canon EOS 5D Mark IV",
                "required": False
            },
            "lens_details": {
                "type": "str",
                "description": "Details about the lens used.",
                "default": "EF 24-70mm f/2.8L II USM",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use camera model and lens details to enhance realism; defaults are set for a common DSLR setup."