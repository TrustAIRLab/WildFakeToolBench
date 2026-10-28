from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SkinPoreDetailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="skin_pore_detailing",
            description="Enhances the realism of images by adding intricate skin details such as pores, freckles, and imperfections."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        skin_details = params.get("skin_details", ["pores", "freckles", "imperfections"])
        detail_intensity = params.get("detail_intensity", 0.5)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_np = np.array(image)

        # Create a texture to simulate skin details
        noise = np.random.normal(0, 1, image_np.shape) * detail_intensity
        noise = np.clip(noise, 0, 255).astype(np.uint8)

        # Apply noise to the image to simulate skin details
        detailed_image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        detailed_image = Image.fromarray(detailed_image_np, "RGB")

        # Optionally apply a slight blur to blend details
        detailed_image = detailed_image.filter(ImageFilter.GaussianBlur(radius=1))

        return detailed_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "skin_details": {
                "type": "list",
                "description": "List of skin details to enhance (e.g., pores, freckles, imperfections).",
                "default": ["pores", "freckles", "imperfections"],
                "required": False
            },
            "detail_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the skin detail effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0.0 (no detail) to 1.0 (maximum detail); useful for enhancing skin realism."