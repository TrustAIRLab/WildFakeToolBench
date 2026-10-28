from typing import Dict, Any
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class ContextualVisualDirectionsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="contextual_visual_directions",
            description="This tool enhances images by applying stylistic filters and effects that mimic recognizable aesthetics, improving brand alignment and authenticity."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        params = {k: v for k, v in params.items() if v is not None}
        self.validate_params(params)
        style = params.get("style", "default")
        theme = params.get("theme", "default")

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Apply effects based on style and theme
        if style == "Wes Anderson":
            img_array = self.apply_wes_anderson_style(img_array)
        elif style == "Nike":
            img_array = self.apply_nike_style(img_array)
        else:
            img_array = self.apply_default_style(img_array)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def apply_wes_anderson_style(self, img_array: np.ndarray) -> np.ndarray:
        # Apply a pastel color palette and vignette effect
        img_array = img_array * np.array([0.9, 0.7, 0.6])  # Desaturate colors
        img_array = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img_array = img_array.filter(ImageFilter.GaussianBlur(radius=2))
        return np.array(img_array)

    def apply_nike_style(self, img_array: np.ndarray) -> np.ndarray:
        # High contrast and sharpness
        img_array = img_array * np.array([1.2, 1.2, 1.2])  # Increase brightness
        img_array = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img_array = img_array.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        return np.array(img_array)

    def apply_default_style(self, img_array: np.ndarray) -> np.ndarray:
        # Mild neutral grade: slight contrast lift and desaturation.
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img = ImageEnhance.Contrast(img).enhance(1.05)
        img = ImageEnhance.Color(img).enhance(0.97)
        return np.array(img)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "style": {
                "type": "str",
                "description": "The visual style to apply (e.g., 'Wes Anderson', 'Nike').",
                "default": "default",
                "required": False
            },
            "theme": {
                "type": "str",
                "description": "The theme to apply (currently not used in effects).",
                "default": "default",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Styles: 'Wes Anderson', 'Nike'; themes are not currently implemented. Use to enhance brand alignment."