"""ComfyUI-H3-FastDerope: a cheaper, motion-gated de-rope for MiniMax H3 fast-motion smear.

Builds on matlowai/ComfyUI-MAINodes (Jerk Oracle, Time Smear, V2V Init, Inject Schedule, Exact Recover).
Drawing Smear / Recover: optional Drawing De-rope (redraw only the 2s drawings of a burst)."""
from .drawing import FastDeropeDrawingRecover, FastDeropeDrawingSmear
from .motion_gate import FastDeropeMotionGate, FastDeropeSegmentSplice
from .tae_loader import FastDeropeTAELoader
from .window_attention import FastDeropeWindowAttention

NODE_CLASS_MAPPINGS = {
    "FastDeropeWindowAttention": FastDeropeWindowAttention,
    "FastDeropeMotionGate": FastDeropeMotionGate,
    "FastDeropeSegmentSplice": FastDeropeSegmentSplice,
    "FastDeropeTAELoader": FastDeropeTAELoader,
    "FastDeropeDrawingSmear": FastDeropeDrawingSmear,
    "FastDeropeDrawingRecover": FastDeropeDrawingRecover,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastDeropeWindowAttention": "H3 Window Attention (fast de-rope pass)",
    "FastDeropeMotionGate": "Motion Gate (de-rope only extreme motion)",
    "FastDeropeSegmentSplice": "Segment Splice (skip pass 2 if calm)",
    "FastDeropeTAELoader": "Load TAE as VAE (taeh3)",
    "FastDeropeDrawingSmear": "Drawing Smear (de-rope only the drawings)",
    "FastDeropeDrawingRecover": "Drawing Recover (hold drawings on 2s/3s)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
