"""
Blur/Sharpen Tool - Applies blur or sharpening filters.
"""

from typing import Dict, Any
from PIL import Image, ImageFilter
from .base_tool import BaseTool


class BlurSharpenTool(BaseTool):
    """Apply blur or sharpen filter to image."""

    def __init__(self):
        super().__init__(
            name="blur_sharpen",
            description="Apply blur (negative values) or sharpen (positive values) filter"
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """
        Apply blur or sharpen to the image.

        Args:
            image: Input PIL Image
            params: {"amount": float (-2.0 to 2.0)}
                   negative = blur, positive = sharpen

        Returns:
            Filtered PIL Image
        """
        self.validate_params(params)
        amount = params.get("amount", -0.5)

        if amount == 0:
            return image.copy()

        filtered_image = image.copy()

        if amount < 0:
            # Apply blur
            # Convert amount to radius (absolute value, scaled)
            radius = abs(amount)
            filtered_image = filtered_image.filter(ImageFilter.GaussianBlur(radius))

        else:
            # Apply sharpen
            # Apply sharpen filter multiple times for stronger effect
            iterations = int(abs(amount))
            for _ in range(iterations):
                filtered_image = filtered_image.filter(ImageFilter.SHARPEN)

            # For fractional part, apply unsharp mask
            if amount % 1.0 > 0:
                filtered_image = filtered_image.filter(
                    ImageFilter.UnsharpMask(radius=1, percent=int(150 * (amount % 1.0)))
                )

        return filtered_image

    def get_guidance(self) -> str:
        return "Applies blur (negative) or sharpen (positive). Blur -0.3 to -0.8 mimics slight camera defocus. Targets 'unnaturally sharp' or 'too crisp edges' feedback."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "amount": {
                "type": "float",
                "min": -2.0,
                "max": 2.0,
                "description": "Blur/sharpen amount (negative=blur, positive=sharpen, 0=no change)",
                "default": -0.5,
                "required": False
            }
        }
