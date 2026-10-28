from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SubjectDistanceOptimizationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="subject_distance_optimization",
            description="Adjusts the perceived distance of subjects in AI-generated images to enhance depth of field effects, creating a more natural blur and realistic appearance."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        subject_distance = params.get("subject_distance", 0.5)
        focus_area = params.get("focus_area", (0.5, 0.5))  # Center of the image by default

        # Convert image to RGB if needed
        image = image.convert("RGB")
        width, height = image.size

        # Create a mask for the focus area
        mask = np.zeros((height, width), dtype=np.float32)
        focus_x, focus_y = int(focus_area[0] * width), int(focus_area[1] * height)
        
        # Create a Gaussian-like distribution around the focus area
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - focus_x) ** 2 + (y - focus_y) ** 2)
                mask[y, x] = np.exp(- (distance ** 2) / (2 * (subject_distance * min(width, height)) ** 2))

        # Normalize the mask to use it for blending
        mask = (mask / mask.max() * 255).astype(np.uint8)

        # Create a blurred version of the image
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))

        # Blend the original and blurred images based on the mask
        blended_image = Image.new("RGB", image.size)
        for y in range(height):
            for x in range(width):
                original_pixel = np.array(image.getpixel((x, y)))
                blurred_pixel = np.array(blurred_image.getpixel((x, y)))
                blend_factor = mask[y, x] / 255.0
                blended_pixel = (1 - blend_factor) * original_pixel + blend_factor * blurred_pixel
                blended_image.putpixel((x, y), tuple(np.clip(blended_pixel, 0, 255).astype(np.uint8)))

        return blended_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "subject_distance": {
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "description": "Controls the perceived distance of the subject; higher values increase blur.",
                "default": 0.5,
                "required": False
            },
            "focus_area": {
                "type": "tuple",
                "description": "Coordinates (x, y) of the focus area as a fraction of image size (0.0 to 1.0).",
                "default": (0.5, 0.5),
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust subject_distance (0.1 to 1.0) and focus_area (0.0 to 1.0) for optimal depth of field effects."