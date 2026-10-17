"""Temporal window attention for MiniMax H3 (built for the de-rope pass).

Why: H3 draws a fast pose crisply when it sits in its own (static) latent token, which is what the de-rope's
time smear arranges, but the regenerated clip is 2.3-3.3x longer and full attention grows with the square of
its length. In that pass the layout and acting already come from the init; a frame mostly needs its temporal
neighbours (where the motion detail is) plus some global context for identity. So each block of video query
frames attends EXACTLY to:
  - every non-video row (text, keyframe/reference conditioning, audio)
  - the video frames within +-window latent frames
  - anchor frames every anchor_stride latent frames (global identity / colour context)
Non-video query rows (audio, text) keep full attention. Cost becomes ~linear in clip length. Every kept key is
exact (no score estimation), unlike block-sparse selection, which can drop the neighbours that carry the motion.
"""
import logging

import torch

import comfy.patcher_extension

log = logging.getLogger("FastDerope")


class _WindowPatch:
    def __init__(self, window, chunk, anchor_stride, min_frames, sigma_hi, sigma_lo):
        self.window, self.chunk, self.anchor_stride = window, chunk, anchor_stride
        self.min_frames, self.sigma_hi, self.sigma_lo = min_frames, sigma_hi, sigma_lo
        self.installed = set()
        self.plans = {}
        self.logged = False

    def plan(self, layout, device):
        """Per chunk: (q_lo, q_hi) video rows and the key row index tensor."""
        key = (layout.seq_len, tuple(layout.segments), str(device))
        p = self.plans.get(key)
        if p is not None:
            return p
        va, vb = next((a, b) for a, b, kind in layout.segments if kind == "video")
        T = int(layout.signature[1])
        F = (vb - va) // T
        prefix = torch.arange(0, va, device=device)
        chunks = []
        for f0 in range(0, T, self.chunk):
            f1 = min(T, f0 + self.chunk)
            lo, hi = max(0, f0 - self.window), min(T, f1 + self.window)
            frames = list(range(lo, hi))
            if self.anchor_stride > 0:
                frames += [f for f in range(0, T, self.anchor_stride) if f < lo or f >= hi]
            frames = sorted(set(frames))
            fr = torch.tensor(frames, device=device)
            rows = (va + fr[:, None] * F + torch.arange(F, device=device)[None]).reshape(-1)
            chunks.append((va + f0 * F, va + f1 * F, torch.cat([prefix, rows])))
        p = {"va": va, "vb": vb, "T": T, "F": F, "chunks": chunks}
        while len(self.plans) >= 4:
            del self.plans[next(iter(self.plans))]
        self.plans[key] = p
        return p


def _make_override(patch, previous):
    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        to = kwargs.get("transformer_options") or {}

        def attend(qq, kk, vv):
            kw = dict(mask=None, attn_precision=attn_precision, skip_reshape=True, skip_output_reshape=False, **kwargs)
            return func(qq, kk, vv, heads, **kw) if previous is None else previous(func, qq, kk, vv, heads, **kw)

        layout = to.get("minimax_h3_layout")
        sigmas = to.get("sigmas")
        ok = (mask is None and skip_reshape and not skip_output_reshape and layout is not None
              and q.shape[2] == layout.seq_len and q.shape == k.shape)
        if ok and sigmas is not None:
            s = float(sigmas.flatten()[0])
            ok = patch.sigma_lo <= s <= patch.sigma_hi
        if ok:
            p = patch.plan(layout, q.device)
            ok = p["T"] >= patch.min_frames
        if not ok:
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
            return func(q, k, v, heads, **kw) if previous is None else previous(func, q, k, v, heads, **kw)

        va = p["va"]
        outs = []
        if va > 0:  # text / conditioning / audio queries: full attention
            outs.append(attend(q[:, :, :va], k, v))
        for q0, q1, idx in p["chunks"]:
            outs.append(attend(q[:, :, q0:q1], k.index_select(2, idx), v.index_select(2, idx)))
        if not patch.logged:
            patch.logged = True
            n_keys = sum(len(c[2]) for c in p["chunks"]) / max(1, len(p["chunks"]))
            log.info(f"WindowAttention: {p['T']} latent frames x {p['F']} rows, {len(p['chunks'])} chunks, "
                     f"~{int(n_keys)} of {q.shape[2]} keys per video query ({n_keys / q.shape[2]:.0%})")
        return torch.cat(outs, dim=1)

    return override


def _install(patch, transformer_options):
    cur = transformer_options.get("optimized_attention_override")
    if cur in patch.installed:
        return
    ov = _make_override(patch, cur)
    patch.installed.add(ov)
    transformer_options["optimized_attention_override"] = ov


class FastDeropeWindowAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "window": ("INT", {"default": 4, "min": 1, "max": 64,
                               "tooltip": "latent frames on each side every video frame attends to exactly (1 latent frame = 4 video frames)"}),
            "chunk": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "query latent frames processed together"}),
            "anchor_stride": ("INT", {"default": 8, "min": 0, "max": 128,
                                      "tooltip": "also attend every Nth latent frame of the whole clip (global identity); 0 = off"}),
            "min_frames": ("INT", {"default": 24, "min": 1, "max": 4096,
                                   "tooltip": "clips with fewer latent frames run dense (short clips gain nothing)"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Temporal window attention for MiniMax H3: each video frame attends exactly to nearby frames, "
                   "anchor frames and all text/audio/conditioning rows. Made for the long de-rope pass 2, whose "
                   "layout comes from the init; cost grows linearly instead of quadratically with clip length.")

    def patch(self, model, window, chunk, anchor_stride, min_frames, start_percent, end_percent):
        ms = model.get_model_object("model_sampling")
        p = _WindowPatch(window, chunk, anchor_stride, min_frames,
                         float(ms.percent_to_sigma(start_percent)), float(ms.percent_to_sigma(end_percent)))
        m = model.clone()
        _install(p, m.model_options["transformer_options"])
        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE, "fast_derope_window_attention",
                                lambda mp, timestep, model_options: _install(p, model_options["transformer_options"]))
        return (m,)
