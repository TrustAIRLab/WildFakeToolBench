from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class SkinTextureOverlayTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="skin_texture_overlay",
            description="Adds a realistic skin-pore texture overlay to AI-generated skin using frequency separation, enhancing realism."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        blend_mode = params.get("blend_mode", "multiply")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_array = np.array(image)

        # Create a synthetic skin texture as a reference
        texture = self.create_skin_texture(image_array.shape[0:2])

        # Apply frequency separation
        high_freq = image_array - gaussian_filter(image_array, sigma=5)
        texture = texture[:image_array.shape[0], :image_array.shape[1], :]

        # Blend the high frequency with the texture
        if blend_mode == "multiply":
            blended = high_freq * (texture / 255.0)
        else:
            blended = high_freq + texture

        blended = np.clip(blended + image_array, 0, 255).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")

    def create_skin_texture(self, shape):
        """Create a synthetic skin texture for overlay."""
        texture = np.random.normal(loc=128, scale=20, size=(shape[0], shape[1], 3)).astype(np.uint8)
        return texture

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "blend_mode": {
                "type": "str",
                "description": "The blending mode to apply the texture overlay.",
                "default": "multiply",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Blend mode options: 'multiply' for realistic texture application. Ideal for enhancing AI-generated skin realism."