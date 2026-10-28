from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SocialPlatformContextTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="social_platform_context",
            description="Applies stylistic adjustments to an image based on selected social media platform aesthetics, influencing lighting, contrast, and sharpness."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        platform = params.get("platform", "instagram").lower()

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply different styles based on the selected platform
        if platform == "instagram":
            image = self.instagram_style(image)
        elif platform == "facebook":
            image = self.facebook_style(image)
        elif platform == "twitter":
            image = self.twitter_style(image)
        elif platform == "snapchat":
            image = self.snapchat_style(image)
        else:
            raise ValueError("Unsupported platform. Choose from: instagram, facebook, twitter, snapchat.")

        return image

    def instagram_style(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast
        image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Slight blur for softness
        return image

    def facebook_style(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # Increase brightness
        return image

    def twitter_style(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)  # Increase sharpness
        return image

    def snapchat_style(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.3)  # Increase color saturation
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "platform": {
                "type": "str",
                "description": "The social media platform to emulate (instagram, facebook, twitter, snapchat).",
                "default": "instagram",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Choose from platforms: instagram, facebook, twitter, snapchat to adjust image style."