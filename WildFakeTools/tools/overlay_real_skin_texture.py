from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class OverlayRealSkinTextureTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="overlay_real_skin_texture",
            description="Overlaying a real high-frequency skin texture on AI-generated images restores natural pores, making the image appear more authentic."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        texture_layer_opacity = params.get("texture_layer_opacity", 0.5)
        texture_scale = params.get("texture_scale", 1.0)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_array = np.array(image)

        # Generate a synthetic skin texture
        texture_height, texture_width = image_array.shape[:2]
        texture = self.generate_skin_texture(texture_height, texture_width, texture_scale)
        
        # Blend the texture with the original image
        blended_image = (1 - texture_layer_opacity) * image_array + texture_layer_opacity * texture
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

        return Image.fromarray(blended_image, mode="RGB")

    def generate_skin_texture(self, height: int, width: int, scale: float) -> np.ndarray:
        # Create a random noise texture
        noise = np.random.normal(loc=0.5, scale=0.1, size=(height, width, 3))
        noise = np.clip(noise, 0, 1)

        # Scale the texture
        texture = np.kron(noise, np.ones((int(scale), int(scale), 1)))
        texture = texture[:height, :width, :]  # Crop to original size
        return (texture * 255).astype(np.uint8)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "texture_layer_opacity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Opacity of the texture layer to blend with the original image.",
                "default": 0.5,
                "required": False
            },
            "texture_scale": {
                "type": "float",
                "min": 0.1,
                "max": 5.0,
                "description": "Scale factor for the skin texture detail.",
                "default": 1.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Texture layer opacity: 0.0 to 1.0; Texture scale: 0.1 to 5.0. Use to enhance realism in AI-generated images."