from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class FaceSkinRefinementTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="face_skin_refinement",
            description="Enhances facial features and smooths skin texture in images, improving realism."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        eye_enhancement = params.get("eye_enhancement", 0.5)
        skin_smoothing = params.get("skin_smoothing", 0.5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_np = np.array(image)

        # Apply skin smoothing
        if skin_smoothing > 0:
            smooth_filter = ImageFilter.GaussianBlur(radius=2 * skin_smoothing)
            image = image.filter(smooth_filter)

        # Convert back to array for eye enhancement
        image_np = np.array(image)

        # Apply eye enhancement by increasing contrast in a specific region
        if eye_enhancement > 0:
            # Define a mask for eye regions (this is a simplistic approach)
            height, width, _ = image_np.shape
            eye_mask = np.zeros((height, width), dtype=np.float32)

            # Mask the eye region so it can be enhanced separately from the rest.
            eye_mask[int(height * 0.3):int(height * 0.4), int(width * 0.35):int(width * 0.65)] = 1

            # Enhance the eye region
            image_np[:, :, 0] = np.clip(image_np[:, :, 0] + (eye_mask * 255 * eye_enhancement), 0, 255)  # Red channel
            image_np[:, :, 1] = np.clip(image_np[:, :, 1] - (eye_mask * 50 * eye_enhancement), 0, 255)   # Green channel
            image_np[:, :, 2] = np.clip(image_np[:, :, 2] - (eye_mask * 50 * eye_enhancement), 0, 255)   # Blue channel

        return Image.fromarray(image_np.astype(np.uint8))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "eye_enhancement": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Enhancement level for eye features.",
                "default": 0.5,
                "required": False
            },
            "skin_smoothing": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Level of skin texture smoothing.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Eye enhancement and skin smoothing levels range from 0 to 1, useful for refining facial details."