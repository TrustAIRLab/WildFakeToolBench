from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class CompressionManipulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="compression_manipulation",
            description="This tool resizes and recompresses the image at a lower JPEG quality, shifting frequency artifacts and degrading the AI fingerprint in the DCT domain."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        resize_factor = params.get("resize_factor", 1.0)
        jpeg_quality = params.get("jpeg_quality", 75)

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Save the image to a bytes buffer with the specified JPEG quality
        from io import BytesIO
        buffer = BytesIO()
        resized_image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)

        # Load the image back from the buffer
        manipulated_image = Image.open(buffer)

        return manipulated_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "resize_factor": {
                "type": "float",
                "min": 0.9,
                "max": 1.0,
                "description": "Factor by which to resize the image (0.9 to 1.0).",
                "default": 1.0,
                "required": False
            },
            "jpeg_quality": {
                "type": "int",
                "min": 50,
                "max": 90,
                "description": "JPEG quality for recompression (50 to 90).",
                "default": 75,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Resize factor (0.9-1.0) and JPEG quality (50-90) can be adjusted to manipulate compression artifacts."