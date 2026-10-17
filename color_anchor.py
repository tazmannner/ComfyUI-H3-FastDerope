"""Color Anchor: keep pass 1's colors in the redraw without keeping its smear.

Pass 2 starts from ~92% noise, so it re-decides details the prompt leaves open; we saw a plaid orange skirt come back
grey. Colors live in the coarse, low-frequency part of the latent; smear lives in fine detail (edges, limb shapes).
During the high-noise steps this patch replaces the coarse part of the model's clean-video prediction (x0) with the
coarse part of the init latent (the slowed pass-1 clip) and leaves all finer detail to the model:
    x0 <- x0 + strength * (LP(init) - LP(x0)),  LP = spatial average over `cell` x `cell` latent cells, per frame.
"""
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("FastDerope")


def _lowpass(v, cell):
    """v: [B, C, T, h, w] -> same shape, spatially averaged over cell x cell latent blocks (smooth upsample)."""
    B, C, T, h, w = v.shape
    x = v.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w).float()
    lo = F.avg_pool2d(x, cell, stride=cell, ceil_mode=True, count_include_pad=False)
    lo = F.interpolate(lo, size=(h, w), mode="bilinear", align_corners=False)
    return lo.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)


class FastDeropeColorAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "init": ("LATENT", {"tooltip": "the encoded slowed clip (VAE Encode output that feeds H3 V2V Init)"}),
            "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                         "tooltip": "how much of pass 1's coarse colour layout is kept in the high-noise steps"}),
            "cell": ("INT", {"default": 4, "min": 1, "max": 16,
                     "tooltip": "coarseness in latent cells (16 px each): 4 = 64 px blocks. Larger keeps only broad colours"}),
            "sigma_low": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "anchor only while sigma >= this (the steps where colours are decided)"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("EXPERIMENTAL. Pass-2 patch: keeps pass 1's coarse colours (latent low frequencies of the slowed clip) during the "
                   "high-noise steps, so the redraw can't change a costume's colour; edges and shapes stay free, so "
                   "the smear is still redrawn crisp. Can leave a faint see-through double on huge, very fast motion.")

    def patch(self, model, init, strength, cell, sigma_low):
        ref = init["samples"]
        if hasattr(ref, "is_nested") and ref.is_nested:
            ref = ref.tensors[0]
        ref = ref.detach()
        m = model.clone()
        state = {"lp": None}

        def post_cfg(args):
            den = args["denoised"]
            sigma = float(args["sigma"].flatten()[0])
            if strength <= 0 or sigma < sigma_low:
                return den
            B = den.shape[0]
            _, C, T, h, w = ref.shape
            n = C * T * h * w
            flat = den.reshape(B, -1)
            if flat.shape[1] < n:
                return den
            v = flat[:, :n].reshape(B, C, T, h, w)
            if state["lp"] is None or state["lp"].device != v.device:
                state["lp"] = _lowpass(ref.to(v.device), cell)
                log.info(f"ColorAnchor: active for sigma >= {sigma_low} (strength {strength}, {cell}x{cell} cells)")
            v_new = v.float() + strength * (state["lp"].expand(B, -1, -1, -1, -1) - _lowpass(v, cell))
            flat = flat.clone()
            flat[:, :n] = v_new.to(flat.dtype).reshape(B, -1)
            return flat.reshape(den.shape)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)
