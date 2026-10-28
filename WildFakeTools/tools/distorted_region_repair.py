from typing import Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class DistortedRegionRepairTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="distorted_region_repair",
            description="This tool uses inpainting techniques to fix distorted or missing areas in AI-generated images, making them appear more natural by correcting common artifacts."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters
        inpainting_strength = params.get("inpainting_strength", 0.5)
        # Default selection is a central box.
        w, h = image.size
        area_selection = params.get(
            "area_selection",
            [(w // 4, h // 4), (3 * w // 4, h // 4),
             (3 * w // 4, 3 * h // 4), (w // 4, 3 * h // 4)],
        )

        # Convert image to numpy array
        image_np = np.array(image.convert("RGB"))

        # Create a mask for inpainting
        mask_image = Image.new("L", (image_np.shape[1], image_np.shape[0]), 0)

        # A polygon mask needs at least two (x, y) points.
        if area_selection is not None and len(area_selection) >= 2:
            points = [tuple(p) for p in area_selection]
            draw = ImageDraw.Draw(mask_image)
            draw.polygon(points, fill=1)

        mask = np.array(mask_image, dtype=np.uint8)

        # Apply Gaussian blur to the mask to create a smooth transition
        mask = gaussian_filter(mask.astype(float), sigma=5)
        mask = np.clip(mask, 0, 1)

        # Inpainting effect: blend the original image with a blurred version based on the mask
        blurred_image = gaussian_filter(image_np.astype(float), sigma=3)
        inpainted_image = (1 - mask[..., np.newaxis]) * image_np + mask[..., np.newaxis] * blurred_image

        # Scale the inpainting strength
        inpainted_image = (1 - inpainting_strength) * image_np + inpainting_strength * inpainted_image

        # Convert back to PIL Image
        return Image.fromarray(np.uint8(np.clip(inpainted_image, 0, 255)))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "inpainting_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the inpainting effect, where 0 means no effect and 1 means full inpainting.",
                "default": 0.5,
                "required": False
            },
            "area_selection": {
                "type": "list",
                "description": "List of (x, y) tuples defining the polygon area to inpaint.",
                "default": None,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use inpainting_strength between 0-1 to control the effect; area_selection should define the area to fix."