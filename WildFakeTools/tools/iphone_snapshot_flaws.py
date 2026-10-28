from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class IphoneSnapshotFlawsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="iphone_snapshot_flaws",
            description="This tool simulates common imperfections found in amateur photos, such as motion blur and awkward framing, to enhance the realism of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        photo_type = params.get("photo_type", "casual")
        imperfection_types = params.get("imperfection_types", ["motion_blur", "color_fade"])

        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply imperfections based on specified types
        if "motion_blur" in imperfection_types:
            image = self.apply_motion_blur(image)

        if "color_fade" in imperfection_types:
            image = self.apply_color_fade(image)

        return image

    def apply_motion_blur(self, image: Image.Image) -> Image.Image:
        # Create a motion blur effect
        blur_radius = np.random.uniform(2, 5)  # Random blur radius
        return image.filter(ImageFilter.GaussianBlur(blur_radius))

    def apply_color_fade(self, image: Image.Image) -> Image.Image:
        # Create a color fade effect
        np_image = np.array(image)
        fade_factor = np.random.uniform(0.7, 1.0)  # Random fade factor
        np_image = np.clip(np_image * fade_factor, 0, 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "photo_type": {
                "type": "str",
                "description": "Type of photo to simulate.",
                "default": "casual",
                "required": False
            },
            "imperfection_types": {
                "type": "list",
                "description": "List of imperfections to apply.",
                "default": ["motion_blur", "color_fade"],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: photo_type (str), imperfection_types (list of str). Use to simulate unremarkable amateur photos."