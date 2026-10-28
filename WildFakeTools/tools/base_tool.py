"""
Base class for image manipulation tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
import numpy as np


class BaseTool(ABC):
    """Abstract base class for all image manipulation tools."""

    def __init__(self, name: str, description: str):
        """
        Initialize a tool.

        Args:
            name: Tool name (e.g., "jpeg_compression")
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """
        Apply the tool to an image.

        Args:
            image: Input PIL Image
            params: Dictionary of tool-specific parameters

        Returns:
            Modified PIL Image
        """
        pass

    @abstractmethod
    def get_param_schema(self) -> Dict[str, Any]:
        """
        Get the schema for tool parameters.

        Returns:
            Dictionary describing expected parameters, types, and ranges
            Example:
            {
                "quality": {
                    "type": "int",
                    "min": 50,
                    "max": 95,
                    "description": "JPEG quality level"
                }
            }
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against schema.

        Args:
            params: Parameters to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        schema = self.get_param_schema()

        for param_name, param_info in schema.items():
            # Check if parameter is required (default: required unless "required": False or has "default")
            is_required = param_info.get("required", True)
            has_default = "default" in param_info

            if param_name not in params:
                # Only error if required and no default
                if is_required and not has_default:
                    raise ValueError(f"Missing required parameter: {param_name}")
                continue  # Skip validation for missing optional params

            value = params[param_name]
            param_type = param_info.get("type")

            # Type checking
            if param_type == "int" and not isinstance(value, int):
                raise ValueError(f"Parameter {param_name} must be int, got {type(value)}")
            elif param_type == "float" and not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {param_name} must be float, got {type(value)}")
            elif param_type == "str" and not isinstance(value, str):
                raise ValueError(f"Parameter {param_name} must be str, got {type(value)}")

            # Range checking (only meaningful for numeric params with a real,
            # comparable bound — some schemas declare tuple/list bounds for
            # vector params, so guard the comparison itself too).
            try:
                if param_info.get("min") is not None and isinstance(value, (int, float)) and value < param_info["min"]:
                    raise ValueError(f"Parameter {param_name} must be >= {param_info['min']}")
                if param_info.get("max") is not None and isinstance(value, (int, float)) and value > param_info["max"]:
                    raise ValueError(f"Parameter {param_name} must be <= {param_info['max']}")
            except TypeError:
                pass

        return True

    def get_guidance(self) -> str:
        """
        Return tool-specific usage guidance.
        Override in subclasses to provide parameter hints.
        """
        return ""

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
