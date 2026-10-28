from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class ShadowConsistencyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="shadow_consistency",
            description="This tool enhances the consistency of shadows in an image by adjusting their direction and intensity, making the image appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        shadow_direction = params.get("shadow_direction", "left")
        shadow_intensity = params.get("shadow_intensity", 5)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create shadow effect based on direction and intensity
        shadow_mask = np.zeros_like(img_array)

        # Define shadow direction vectors
        direction_map = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, -1),
            "down": (0, 1)
        }

        if shadow_direction not in direction_map:
            raise ValueError("Invalid shadow direction. Choose from 'left', 'right', 'up', 'down'.")

        dx, dy = direction_map[shadow_direction]
        shadow_intensity = np.clip(shadow_intensity, 1, 10)

        # Apply shadow effect
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                if 0 <= x + dx < img_array.shape[1] and 0 <= y + dy < img_array.shape[0]:
                    shadow_mask[y + dy, x + dx] = img_array[y, x] * (0.5 * shadow_intensity / 10)

        # Blend original image with shadow mask
        blended_image = np.clip(img_array + shadow_mask, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_image, "RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "shadow_direction": {
                "type": "str",
                "description": "Direction of the shadows (left, right, up, down).",
                "default": "left",
                "required": False
            },
            "shadow_intensity": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Intensity of the shadows, where 1 is very light and 10 is very dark.",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust shadow direction (left, right, up, down) and intensity (1-10) to enhance realism in AI-generated images."