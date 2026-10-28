from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class AmateurSnapshotAestheticTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="amateur_snapshot_aesthetic",
            description="Applies a casual, amateur snapshot aesthetic to an image, simulating the look of smartphone photos with a soft focus and slight color adjustments."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        keywords = params.get("keywords", "smartphone photo, candid shot")

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply a slight blur to simulate amateur photography
        image = image.filter(ImageFilter.GaussianBlur(radius=1))

        # Convert to numpy array for color manipulation
        img_array = np.array(image)

        # Adjust colors slightly to mimic the casual snapshot feel
        img_array = img_array.astype(np.float32)
        img_array[..., 0] = np.clip(img_array[..., 0] * 1.1, 0, 255)  # Increase red channel
        img_array[..., 1] = np.clip(img_array[..., 1] * 0.95, 0, 255)  # Decrease green channel
        img_array[..., 2] = np.clip(img_array[..., 2] * 0.9, 0, 255)   # Decrease blue channel
        img_array = img_array.astype(np.uint8)

        # Convert back to PIL Image
        result_image = Image.fromarray(img_array)

        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "keywords": {
                "type": "str",
                "description": "Keywords to influence the aesthetic style, e.g., 'smartphone photo, candid shot'.",
                "default": "smartphone photo, candid shot",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use keywords to adjust the aesthetic; suitable for creating a casual snapshot style."