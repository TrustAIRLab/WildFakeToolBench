from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class LightingColorHarmonizationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="lighting_color_harmonization",
            description="Adjusts the lighting, shadows, and color tone of an image to match a real background, creating a seamless integration."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # background_color/color_tone accept "R,G,B" strings or numeric triples.
        def _parse_rgb(value, default):
            if isinstance(value, str):
                try:
                    parts = [float(v) for v in value.split(",")]
                    return parts if len(parts) == 3 else default
                except ValueError:
                    return default
            return value

        background_color = _parse_rgb(params.get("background_color", (255, 255, 255)), (255, 255, 255))
        shadow_intensity = params.get("shadow_intensity", 0.5)
        color_tone = _parse_rgb(params.get("color_tone", (255, 255, 255)), (255, 255, 255))

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image, dtype=np.float32)

        # Calculate the average color of the image
        avg_color = img_array.mean(axis=(0, 1))

        # Adjust the image colors towards the background color
        color_adjustment = np.array(background_color) - avg_color
        img_array += color_adjustment * (1 - shadow_intensity)

        # Apply color tone adjustment
        img_array = np.clip(img_array * (np.array(color_tone) / 255.0), 0, 255)

        # Convert back to PIL Image
        result_image = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "background_color": {
                "type": "str",
                "description": "Background color in RGB format (e.g., '255,255,255')",
                "default": "255,255,255",
                "required": False
            },
            "shadow_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the shadow effect (0.0 to 1.0)",
                "default": 0.5,
                "required": False
            },
            "color_tone": {
                "type": "str",
                "description": "Color tone adjustment in RGB format (e.g., '255,255,255')",
                "default": "255,255,255",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: background_color (RGB), shadow_intensity (0.0 to 1.0), color_tone (RGB). Use when harmonizing AI subjects with real backgrounds."