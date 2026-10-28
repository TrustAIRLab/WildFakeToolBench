from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool

SUPPORTED_TIMEFRAMES = ("vintage", "modern", "film", "polaroid")
DEFAULT_TIMEFRAME = "film"


class TimeframeSpecificationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="timeframe_specification",
            description="Adjusts the quality and style of an image to reflect the characteristics of photos from a specified era."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        timeframe = str(params.get("timeframe", DEFAULT_TIMEFRAME)).lower()
        if timeframe not in SUPPORTED_TIMEFRAMES:
            # Unrecognised values use the default timeframe.
            timeframe = DEFAULT_TIMEFRAME
        
        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply effects based on the specified timeframe
        if timeframe == "vintage":
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
            image = self.apply_vintage_tint(image)
        elif timeframe == "modern":
            pass  # No changes for modern
        elif timeframe == "film":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            image = self.add_film_grain(image)
        elif timeframe == "polaroid":
            image = self.apply_polaroid_effect(image)

        return image

    def apply_vintage_tint(self, image: Image.Image) -> Image.Image:
        """Apply a vintage tint to the image."""
        np_image = np.array(image)
        # Apply a sepia-like filter
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                  [0.349, 0.686, 0.168],
                                  [0.272, 0.534, 0.131]])
        tinted_image = np.clip(np_image @ sepia_filter.T, 0, 255).astype(np.uint8)
        return Image.fromarray(tinted_image)

    def add_film_grain(self, image: Image.Image) -> Image.Image:
        """Add film grain effect to the image."""
        np_image = np.array(image)
        noise = np.random.normal(0, 4, np_image.shape)
        noisy_image = np.clip(np_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def apply_polaroid_effect(self, image: Image.Image) -> Image.Image:
        """Simulate a Polaroid effect by adding a white border."""
        border_size = 20
        np_image = np.array(image)
        bordered_image = np.ones((np_image.shape[0] + border_size, np_image.shape[1] + border_size, 3), dtype=np.uint8) * 255
        bordered_image[border_size//2:border_size//2 + np_image.shape[0], border_size//2:border_size//2 + np_image.shape[1]] = np_image
        return Image.fromarray(bordered_image)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "timeframe": {
                "type": "str",
                "description": "Timeframe to emulate: vintage, modern, film or polaroid. 'modern' is a pass-through; unrecognised values fall back to the default.",
                "default": DEFAULT_TIMEFRAME,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Specify a timeframe (vintage, modern, film, polaroid) to adjust the image style and quality."