from typing import Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from .base_tool import BaseTool


class AnalogArtifactsOverlayTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="analog_artifacts_overlay",
            description="Adds overlays of dust, scratches, and light leaks to images, mimicking the imperfections of analog film."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        overlay_type = params.get("overlay_type", "dust")
        overlay_opacity = params.get("overlay_opacity", 0.5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        width, height = image.size

        # Create an overlay
        overlay = Image.new("RGBA", image.size)

        if overlay_type == "dust":
            self.add_dust_overlay(overlay)
        elif overlay_type == "scratch":
            self.add_scratch_overlay(overlay)
        elif overlay_type == "light leak":
            self.add_light_leak_overlay(overlay)

        # Composite the overlay with the original image
        overlay = overlay.convert("RGB")
        combined = Image.blend(image, overlay, overlay_opacity)
        return combined

    def add_dust_overlay(self, overlay: Image.Image):
        draw = ImageDraw.Draw(overlay)
        for _ in range(100):  # Add 100 dust particles
            x = np.random.randint(0, overlay.width)
            y = np.random.randint(0, overlay.height)
            radius = np.random.randint(1, 5)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 255, 255, np.random.randint(50, 150)))

    def add_scratch_overlay(self, overlay: Image.Image):
        draw = ImageDraw.Draw(overlay)
        for _ in range(10):  # Add 10 scratches
            x1 = np.random.randint(0, overlay.width)
            y1 = np.random.randint(0, overlay.height)
            x2 = np.random.randint(0, overlay.width)
            y2 = np.random.randint(0, overlay.height)
            draw.line((x1, y1, x2, y2), fill=(255, 255, 255, np.random.randint(50, 100)), width=np.random.randint(1, 3))

    def add_light_leak_overlay(self, overlay: Image.Image):
        draw = ImageDraw.Draw(overlay)
        for _ in range(3):  # Add 3 light leaks
            x = np.random.randint(0, overlay.width)
            y = np.random.randint(0, overlay.height)
            width = np.random.randint(100, 300)
            height = np.random.randint(100, 300)
            draw.rectangle((x, y, x + width, y + height), fill=(255, 255, 255, np.random.randint(50, 150)))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "overlay_type": {
                "type": "str",
                "description": "Type of overlay to apply: 'dust', 'scratch', or 'light leak'.",
                "default": "dust",
                "required": False
            },
            "overlay_opacity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Opacity of the overlay effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Overlay types: 'dust', 'scratch', 'light leak'; Opacity range: 0.0 to 1.0."