"""Lazy loader for the tools that call a diffusion image-editing model.

The tools listed in :data:`DIFFUSION_TOOLS` call Qwen-Image-Edit-2511, which needs
torch, diffusers, a CUDA GPU and a local checkpoint. :func:`load_pipeline` reports
whichever of those is missing and how to install it.
"""

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

DIFFUSION_TOOLS = (
    "background_realism",
    "surface_naturalize",
    "iterative_denoise_pass",
)


class DiffusionBackendUnavailable(RuntimeError):
    """Raised when a diffusion-backed tool is used without its backend."""


def _explain(tool_name, model_path, missing):
    return (
        f"Tool {tool_name!r} needs a diffusion backend, which is not installed by "
        f"default.\n\n"
        f"  Missing: {missing}\n\n"
        f"This tool calls the {MODEL_ID} image-editing model.\n\n"
        f"To enable it:\n"
        f"    pip install \"torch>=2.3\" \"diffusers>=0.37\" transformers accelerate\n"
        f"    huggingface-cli download {MODEL_ID} --local-dir {model_path}\n"
        f"  A CUDA GPU is also required.\n\n"
        f"To skip the tools that need it instead:\n"
        f"    from tools import available_tools, get_tool_by_name, DIFFUSION_TOOLS\n"
        f"    tools = [get_tool_by_name(n) for n in available_tools()\n"
        f"             if n not in DIFFUSION_TOOLS]"
    )


def load_pipeline(tool_name, model_path):
    """Return a ready QwenImageEditPlusPipeline, or raise a self-explaining error."""
    try:
        import torch
    except ImportError as exc:
        raise DiffusionBackendUnavailable(
            _explain(tool_name, model_path, f"torch is not installed ({exc})")
        ) from exc

    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        raise DiffusionBackendUnavailable(
            _explain(tool_name, model_path, f"diffusers >= 0.37 is not installed ({exc})")
        ) from exc

    if not torch.cuda.is_available():
        raise DiffusionBackendUnavailable(
            _explain(tool_name, model_path, "no CUDA device is visible to torch")
        )

    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
    except Exception as exc:
        raise DiffusionBackendUnavailable(
            _explain(
                tool_name,
                model_path,
                f"the checkpoint could not be loaded from {model_path!r} ({exc})",
            )
        ) from exc

    return pipeline.to("cuda")
