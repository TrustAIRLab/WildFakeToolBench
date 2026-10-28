from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class AuthenticCompressionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="authentic_compression",
            description="Applies authentic compression that mimics in-camera JPEG processing, introducing typical artifacts and noise found in real images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        compression_quality = params.get("compression_quality", 75)
        compression_type = params.get("compression_type", "JPEG").upper()

        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save the image to a bytes buffer with the specified compression
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format=compression_type, quality=compression_quality)
        buffer.seek(0)

        # Load the image back from the buffer
        compressed_image = Image.open(buffer)

        return compressed_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "compression_quality": {
                "type": "int",
                "min": 0,
                "max": 100,
                "description": "Quality of the compression, where 100 is lossless and 0 is maximum compression.",
                "default": 75,
                "required": False
            },
            "compression_type": {
                "type": "str",
                "min": None,
                "max": None,
                "description": "Type of compression to apply, either 'JPEG' or 'PNG'.",
                "default": "JPEG",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Compression quality (0-100) and type ('JPEG' or 'PNG') can be adjusted to mimic camera outputs."