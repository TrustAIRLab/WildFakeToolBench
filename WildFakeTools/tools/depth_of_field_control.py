from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class DepthOfFieldControlTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="depth_of_field_control",
            description="Simulates shallow or deep focus effects in images, enhancing realism by guiding viewer attention and mimicking natural photographic depth of field."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        depth_of_field = params.get("depth_of_field", "shallow")
        focus_area = params.get("focus_area", (0.5, 0.5))  # Default to center of the image
        focus_radius = params.get("focus_radius", 0.2)  # Default radius for shallow focus

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        focus_x, focus_y = int(focus_area[0] * width), int(focus_area[1] * height)

        # Create a mask for the focus area
        mask = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - focus_x) ** 2 + (y - focus_y) ** 2)
                if depth_of_field == "shallow":
                    mask[y, x] = max(0, 1 - (distance / (focus_radius * min(width, height))))
                else:  # deep
                    mask[y, x] = min(1, distance / (focus_radius * min(width, height)))

        # Apply Gaussian blur based on the mask
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
        final_image = Image.new("RGB", image.size)

        for y in range(height):
            for x in range(width):
                final_image.putpixel((x, y), tuple(
                    int((1 - mask[y, x]) * image.getpixel((x, y))[i] + mask[y, x] * blurred_image.getpixel((x, y))[i]) for i in range(3)
                ))

        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "depth_of_field": {
                "type": "str",
                "description": "Type of depth of field effect: 'shallow' or 'deep'.",
                "default": "shallow",
                "required": False
            },
            "focus_area": {
                "type": "list",
                "description": "Coordinates of the focus area as a fraction of the image size (x, y).",
                "default": [0.5, 0.5],
                "required": False
            },
            "focus_radius": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Radius for the focus area effect, relative to the image size.",
                "default": 0.2,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust depth_of_field for shallow or deep focus; focus_area as (x, y) coordinates; focus_radius to control the size of the focus area."