from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class OpticalImperfectionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="optical_imperfections",
            description="Introduce barrel distortion, vignetting, and chromatic aberration to AI images, making them appear more like actual photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        distortion_strength = params.get("distortion_strength", 0.5)
        vignetting_amount = params.get("vignetting_amount", 0.5)
        chromatic_aberration_strength = params.get("chromatic_aberration_strength", 0.5)

        image = image.convert("RGB")
        img_array = np.array(image)

        # Apply barrel distortion
        height, width = img_array.shape[:2]
        center_x, center_y = width / 2, height / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)

        y_indices, x_indices = np.indices((height, width))
        x_distorted = (x_indices - center_x) * (1 + distortion_strength * ((np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2) / max_radius)**2))
        y_distorted = (y_indices - center_y) * (1 + distortion_strength * ((np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2) / max_radius)**2))

        x_distorted = np.clip(x_distorted + center_x, 0, width - 1).astype(np.float32)
        y_distorted = np.clip(y_distorted + center_y, 0, height - 1).astype(np.float32)

        distorted_image = np.zeros_like(img_array)
        distorted_image[y_indices, x_indices] = img_array[np.clip(y_distorted, 0, height - 1).astype(int), np.clip(x_distorted, 0, width - 1).astype(int)]

        # Apply vignetting
        vignette = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                vignette[i, j] = 1 - (distance / max_radius) * vignetting_amount

        vignette = np.clip(vignette, 0, 1)
        vignetting_image = (distorted_image * vignette[:, :, np.newaxis]).astype(np.uint8)

        # Apply chromatic aberration
        ca_red = np.roll(vignetting_image, int(chromatic_aberration_strength * 5), axis=1)  # Shift red channel right
        ca_blue = np.roll(vignetting_image, -int(chromatic_aberration_strength * 5), axis=1)  # Shift blue channel left

        final_image = (vignetting_image.astype(np.int32) + ca_red.astype(np.int32) + ca_blue.astype(np.int32)) // 3
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

        return Image.fromarray(final_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "distortion_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of barrel distortion effect.",
                "default": 0.5,
                "required": False
            },
            "vignetting_amount": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Amount of vignetting effect.",
                "default": 0.5,
                "required": False
            },
            "chromatic_aberration_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of chromatic aberration effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters range from 0.0 to 1.0; useful for adding realistic optical imperfections to AI-generated images."