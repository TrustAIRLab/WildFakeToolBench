"""
JPEG Compression Tool - Adds compression artifacts to images.
"""

from typing import Dict, Any
from PIL import Image
import io
from .base_tool import BaseTool


class JPEGCompressionTool(BaseTool):
    """Apply JPEG compression to add realistic compression artifacts."""

    def __init__(self):
        super().__init__(
            name="jpeg_compression",
            description="Apply JPEG compression to add realistic compression artifacts"
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """
        Apply JPEG compression with specified quality.

        Args:
            image: Input PIL Image
            params: {"quality": int (50-95)}

        Returns:
            Compressed PIL Image
        """
        self.validate_params(params)
        quality = params.get("quality", 80)

        # Convert to RGB if needed (JPEG doesn't support transparency)
        if image.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Save to bytes buffer with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Load back from buffer
        compressed_image = Image.open(buffer)

        return compressed_image

    def get_guidance(self) -> str:
        return "Effective for adding camera-like compression signature. Quality 75-85 for noticeable effect, 85-92 for subtle. Often works as a finishing tool after semantic edits."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "quality": {
                "type": "int",
                "min": 50,
                "max": 95,
                "description": "JPEG quality level (lower = more compression artifacts)",
                "default": 80,
                "required": False
            }
        }
