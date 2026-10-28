"""
Color Adjustment Tool - Adjusts brightness, contrast, and saturation.
"""

from typing import Dict, Any
from PIL import Image, ImageEnhance
from .base_tool import BaseTool


class ColorAdjustTool(BaseTool):
    """Adjust image brightness, contrast, and saturation."""

    def __init__(self):
        super().__init__(
            name="color_adjust",
            description="Adjust image brightness, contrast, and saturation"
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """
        Adjust color properties of the image.

        Args:
            image: Input PIL Image
            params: {
                "brightness": float (-0.2 to 0.2),
                "contrast": float (0.8 to 1.2),
                "saturation": float (0.8 to 1.2)
            }

        Returns:
            Adjusted PIL Image
        """
        self.validate_params(params)

        brightness_factor = 1.0 + params.get("brightness", 0.03)
        contrast_factor = params.get("contrast", 1.05)
        saturation_factor = params.get("saturation", 0.97)

        adjusted_image = image.copy()

        # Apply brightness
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = enhancer.enhance(brightness_factor)

        # Apply contrast
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(adjusted_image)
            adjusted_image = enhancer.enhance(contrast_factor)

        # Apply saturation (only if image has color)
        if adjusted_image.mode in ("RGB", "RGBA") and saturation_factor != 1.0:
            enhancer = ImageEnhance.Color(adjusted_image)
            adjusted_image = enhancer.enhance(saturation_factor)

        return adjusted_image

    def get_guidance(self) -> str:
        return "Adjusts brightness, contrast, saturation. Use small values (±0.1) for subtle shifts. Effective for 'unnatural colors' or 'over-saturated' feedback."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "brightness": {
                "type": "float",
                "min": -0.2,
                "max": 0.2,
                "default": 0.0,
                "required": False,
                "description": "Brightness adjustment (-0.2=darker, +0.2=brighter, 0.0=no change)"
            },
            "contrast": {
                "type": "float",
                "min": 0.8,
                "max": 1.2,
                "default": 1.0,
                "required": False,
                "description": "Contrast adjustment (1.0=no change)"
            },
            "saturation": {
                "type": "float",
                "min": 0.8,
                "max": 1.2,
                "default": 1.0,
                "required": False,
                "description": "Saturation adjustment (1.0=no change)"
            }
        }
