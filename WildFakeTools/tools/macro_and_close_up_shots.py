from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class MacroAndCloseUpShotsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="macro_and_close_up_shots",
            description="Enhances image details by simulating macro or close-up perspectives, making the image appear more lifelike."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        perspective_type = params.get("perspective_type", "macro")
        detail_level = params.get("detail_level", 5)

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create a blurred version of the image to simulate depth of field
        blur_radius = max(0, min(10, 10 - detail_level))  # Inverse relationship
        blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # Create a mask for the sharp area based on perspective type
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        if perspective_type == "macro":
            # Simulate a macro effect by focusing on the center
            center_x, center_y = image.width // 2, image.height // 2
            mask[center_y-50:center_y+50, center_x-50:center_x+50] = 255
        else:  # close-up
            # Simulate a close-up effect by focusing on a larger area
            mask[image.height//4:image.height*3//4, image.width//4:image.width*3//4] = 255

        # Convert mask to PIL Image
        mask_image = Image.fromarray(mask)

        # Composite the sharp and blurred images
        sharp_image = Image.composite(image, blurred_image, mask_image)

        return sharp_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "perspective_type": {
                "type": "str",
                "description": "Type of perspective to simulate: 'macro' or 'close-up'.",
                "default": "macro",
                "required": False
            },
            "detail_level": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Level of detail enhancement (1 = least, 10 = most).",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use detail_level from 1 to 10; perspective_type can be 'macro' or 'close-up' for enhancing image realism."