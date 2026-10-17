"""Absolute fast-motion measure for gating the de-rope.

MAINodes' Jerk Oracle ranks a clip against itself (phase-normalised profile, quantile threshold), so it always
marks the top ~25% of every clip as "too fast" -- including talking heads and shot cuts. Here motion is measured
in absolute terms: dense optical flow on the (cheaply decoded) pass-1 frames, camera motion removed (median flow),
and the 98th percentile of the remaining per-pixel speed, as a fraction of frame width per frame.
Shot cuts (whole-frame appearance change) are detected and never count as motion."""
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

WORK_W = 256


def _gray(frames):
    """frames: float [F,H,W,3] in 0..1 (ComfyUI IMAGE) or uint8 -> list of uint8 gray at WORK_W."""
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        h, w = a.shape[:2]
        a = cv2.resize(a, (WORK_W, max(8, int(round(h * WORK_W / w)))), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))
    return out


def measure(frames):
    """Returns (speed[F], cut[F]): speed[i] = motion between frame i-1 and i (fraction of width per frame),
    cut[i] = True if frame i starts a new shot. speed[0] = 0."""
    if cv2 is None:
        raise RuntimeError("Motion Gate needs OpenCV: pip install opencv-python-headless (or opencv-python)")
    g = _gray(frames)
    n = len(g)
    speed = np.zeros(n, np.float32)
    cut = np.zeros(n, bool)
    hist = [cv2.calcHist([x], [0], None, [32], [0, 256]).ravel() for x in g]
    hist = [h / (h.sum() + 1e-6) for h in hist]
    for i in range(1, n):
        a, b = g[i - 1], g[i]
        # shot cut: appearance changes wholesale (histogram distance + low correlation)
        hd = 0.5 * np.abs(hist[i] - hist[i - 1]).sum()
        corr = np.corrcoef(a.ravel().astype(np.float32), b.ravel().astype(np.float32))[0, 1]
        if hd > 0.35 or corr < 0.35:
            cut[i] = True
            continue
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        flow -= np.median(flow.reshape(-1, 2), axis=0)          # remove camera pan
        mag = np.linalg.norm(flow, axis=2)
        speed[i] = float(np.percentile(mag, 98)) / a.shape[1]
    # A real shot cut is an isolated wholesale change. Clustered ones are explosions / flashes / whip pans inside
    # an action burst: count those as extreme motion instead.
    cand = np.where(cut)[0]
    for i in cand:
        if np.any((np.abs(cand - i) <= 3) & (cand != i)):
            cut[i] = False
            speed[i] = max(speed[i], 0.06)
    return speed, cut


def summarize(speed, cut, thr):
    hot = (speed > thr) & ~cut
    return {"p50": float(np.median(speed)), "p90": float(np.percentile(speed, 90)), "max": float(speed.max()),
            "hot_frames": int(hot.sum()), "cuts": [int(i) for i in np.where(cut)[0]]}


# ----------------------------------------------------------------------------------------- nodes
def _runs(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - s))
            s = None
    return out


def hot_frames(speed, cut, thr, min_windows, margin):
    """Frames inside sustained fast motion: 4-frame windows (one H3 latent token) whose summed speed > thr,
    kept only in runs of >= min_windows windows, grown by margin frames. Shot cuts never count."""
    s = speed.copy()
    s[cut] = 0.0
    n = len(s)
    w = np.convolve(s, np.ones(4), "valid") if n >= 4 else np.zeros(0)
    hot = np.zeros(n, bool)
    for a, ln in _runs(w > thr):
        if ln >= min_windows:
            hot[a:a + ln + 3] = True
    if margin and hot.any():
        idx = np.where(hot)[0]
        for i in idx:
            hot[max(0, i - margin):min(n, i + margin + 1)] = True
    return hot


def plan_segment(hot, d_max, handle, oracle_holds=None, cuts=()):
    """Hold map for the cropped segment, handles at hold 1. Inside hot spans: the oracle's own per-frame holds
    (min 2) when given, else d_max; one-step-per-4-frames ramps outside them.
    Returns (a, b) inclusive crop bounds and the per-frame holds of the segment, or None if nothing is hot."""
    n = len(hot)
    if not hot.any():
        return None
    idx = np.where(hot)[0]
    dist = np.full(n, 10 ** 6)
    for i in idx:
        dist = np.minimum(dist, np.abs(np.arange(n) - i))
    peak = np.full(n, d_max)
    if oracle_holds is not None and len(oracle_holds) == n:
        peak = np.clip(np.asarray(oracle_holds), 2, d_max)
    holds = np.where(hot, peak, np.maximum(1, d_max - np.ceil(dist / 4.0))).astype(int)
    holds = np.where(hot, holds, np.minimum(holds, 2)).astype(int)  # short shoulders only
    held = np.where(holds > 1)[0]
    a, b = max(0, held[0] - handle), min(n - 1, held[-1] + handle)
    # A shot cut inside a handle: start/end the segment exactly ON the cut. The switch between pass 1 and the redraw
    # then hides in the cut (hard switch, no crossfade), instead of crossfading two different takes of the new shot.
    cin = [int(c) for c in cuts if a < c <= held[0]]
    cout = [int(c) for c in cuts if held[-1] < c <= b]
    if cin:
        a = max(cin)
    if cout:
        b = min(cout) - 1
    return int(a), int(b), [int(h) for h in holds[a:b + 1]], bool(cin), bool(cout)


class FastDeropeMotionGate:
    """Decide WHERE to de-rope from absolute motion, not the oracle's per-clip ranking."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "pass-1 frames (a cheap taeh3 decode is enough)"}),
            "threshold": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 2.0, "step": 0.005,
                          "tooltip": "motion per 4-frame window (one H3 latent token), in frame widths, camera motion removed. "
                                     "Calibrated: dialogue/dance <= 0.08, punches/crashes/fights 0.1-0.6"}),
            "min_windows": ("INT", {"default": 5, "min": 1, "max": 64, "tooltip": "fast motion must last this many consecutive windows"}),
            "margin": ("INT", {"default": 4, "min": 0, "max": 48, "tooltip": "frames added around each hot span"}),
            "d_max": ("INT", {"default": 4, "min": 2, "max": 8, "tooltip": "slow-down factor on hot frames"}),
            "handle_frames": ("INT", {"default": 12, "min": 2, "max": 48, "tooltip": "real-time context kept on each side of the burst"}),
        }, "optional": {
            "freeze_frames": ("INT", {"default": 8, "min": 0, "max": 48,
                              "tooltip": "outer handle frames to lock to pass 1 in pass 2 (needs Freeze Mask -> V2V Init); "
                                         "the splice crossfade then never blends two different takes"}),
            "oracle_hold_map": ("STRING", {"forceInput": True, "tooltip": "H3 Jerk Oracle hold_map: its per-frame slow-down is used inside the hot spans"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("segment_images", "hold_map", "splice_map", "segment_start", "report")
    FUNCTION = "gate"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Finds sustained EXTREME motion in pass 1 (optical flow, camera motion and shot cuts excluded) and "
                   "plans a de-rope for just that span: cropped frames + hold map for H3 Time Smear, and a splice map for "
                   "Anime Cadence Segment Splice. Calm clips (dialogue, gentle motion) come out with active=false and the "
                   "splice node then skips pass 2 entirely.")

    def gate(self, images, threshold, min_windows, margin, d_max, handle_frames, oracle_hold_map=None, freeze_frames=8):
        import json
        frames = images.detach().cpu().numpy()
        speed, cut = measure(frames)
        hot = hot_frames(speed, cut, threshold, min_windows, margin)
        oh = json.loads(oracle_hold_map)["holds"] if oracle_hold_map else None
        plan = plan_segment(hot, d_max, handle_frames, oh, np.where(cut)[0])
        n = frames.shape[0]
        runs = [(int(a), int(a + ln - 1)) for a, ln in _runs(hot)]
        if plan is None:
            splice = {"active": False, "world_len": n}
            rep = (f"MotionGate: nothing extreme (max window {float(np.convolve(speed * ~cut, np.ones(4), 'valid').max()):.3f} "
                   f"vs threshold {threshold}); pass 2 will be skipped. cuts at {np.where(cut)[0].tolist()}")
            return (images[:1], json.dumps({"holds": [1], "world_len": 1}), json.dumps(splice), 0, rep)
        a, b, holds, hard_in, hard_out = plan
        held = [i for i, h in enumerate(holds) if h > 1]
        h_in, h_out = held[0], len(holds) - 1 - held[-1]
        # Outer handle frames to lock to pass 1 inside pass 2 (wire FastDerope Freeze Mask -> H3 V2V Init): the splice
        # crossfade then blends pass 1 with (nearly) itself, never with a different take. Keep >= 4 live handle frames.
        f_in = 0 if hard_in else max(0, min(freeze_frames, h_in - 4))
        f_out = 0 if hard_out else max(0, min(freeze_frames, h_out - 4))
        freeze = [1 if (i < f_in or i >= len(holds) - f_out) else 0 for i in range(len(holds))]
        splice = {"active": True, "start": a, "end": b, "world_len": n, "handle_in": h_in, "handle_out": h_out,
                  "hard_in": hard_in, "hard_out": hard_out, "freeze_in": f_in, "freeze_out": f_out}
        rep = (f"MotionGate: hot frames {runs}; regenerate f{a}-f{b} ({b - a + 1} of {n} frames, "
               f"{sum(holds)} slowed frames); cuts at {np.where(cut)[0].tolist()}; edges: "
               f"{'cut' if hard_in else f'freeze {f_in}'} / {'cut' if hard_out else f'freeze {f_out}'}")
        print("[FastDerope] " + rep)
        # "hot" marks the sustained-fast frames themselves (not shoulders/handles): Drawing Smear redraws only those as drawings
        seg_map = {"holds": holds, "world_len": len(holds), "hot": [int(x) for x in hot[a:b + 1]], "freeze": freeze}
        return (images[a:b + 1], json.dumps(seg_map), json.dumps(splice), a, rep)


class FastDeropeSegmentSplice:
    """Paste the regenerated burst back into the full pass-1 render; skip pass 2 when the gate found nothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "baseline": ("IMAGE", {"tooltip": "full-quality pass-1 frames"}),
            "splice_map": ("STRING", {"forceInput": True}),
            "feather_frames": ("INT", {"default": 6, "min": 0, "max": 24, "tooltip": "crossfade inside each handle"}),
        }, "optional": {
            "frozen": ("BOOLEAN", {"default": True, "tooltip": "on when Freeze Mask locks the outer handle frames in pass 2: "
                       "the crossfade is kept inside those frames"}),
            "restore_static": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                               "tooltip": "same threshold as Freeze Mask static_threshold: pastes pass 1's own pixels back on "
                                          "regions that don't move (full detail, no background boil). 0 = off"}),
            "segment": ("IMAGE", {"lazy": True, "tooltip": "H3 Exact Recover output of the de-roped segment"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "splice"
    CATEGORY = "FastDerope"

    def check_lazy_status(self, baseline, splice_map, feather_frames, segment=None, frozen=True, restore_static=0.0):
        import json
        if json.loads(splice_map).get("active") and segment is None:
            return ["segment"]
        return []

    def splice(self, baseline, splice_map, feather_frames, segment=None, frozen=True, restore_static=0.0):
        import json
        import torch
        sp = json.loads(splice_map)
        out = baseline.detach().float().cpu().clone()
        if not sp.get("active") or segment is None:
            return (out,)
        seg = segment.detach().float().cpu()
        a, b = sp["start"], sp["end"]
        n = b - a + 1
        assert seg.shape[0] == n, (seg.shape, n)
        if seg.shape[1:] != out.shape[1:]:
            seg = torch.nn.functional.interpolate(seg.permute(0, 3, 1, 2), size=out.shape[1:3], mode="bilinear",
                                                  align_corners=False).permute(0, 2, 3, 1)
        fi = min(feather_frames, sp.get("handle_in", feather_frames))
        fo = min(feather_frames, sp.get("handle_out", feather_frames))
        if sp.get("hard_in"):
            fi = 0                                   # segment starts on a shot cut: switch on the cut
        if sp.get("hard_out"):
            fo = 0
        if sp.get("freeze_in") and frozen:
            fi = min(fi, sp["freeze_in"])            # blend only where the redraw was locked to pass 1
        if sp.get("freeze_out") and frozen:
            fo = min(fo, sp["freeze_out"])
        keep = None
        if restore_static > 0:
            st = static_cells(out[a:b + 1], restore_static)                         # [n,gh,gw] 1 = static in pass 1
            keep = torch.nn.functional.interpolate(st[:, None], size=out.shape[1:3], mode="bilinear",
                                                   align_corners=False)[:, 0, :, :, None]
            keep = torch.nn.functional.avg_pool2d(keep[..., 0][:, None], 9, 1, 4)[:, 0, :, :, None]  # soft edge
        for i in range(n):
            w = 1.0
            if fi and i < fi:
                w = (i + 1) / (fi + 1)
            j = n - 1 - i
            if fo and j < fo:
                w = min(w, (j + 1) / (fo + 1))
            if keep is not None:
                w = w * (1.0 - keep[i])
            out[a + i] = out[a + i] * (1 - w) + seg[i] * w
        return (out,)


def static_cells(frames, thr, grow=1, cell=16):
    """frames: [T,H,W,C] float 0..1 (torch). Returns [T, H//cell, W//cell] float, 1 = region that does not move within
    +-2 frames (max abs grey change, averaged per cell, below thr), shrunk by `grow` cells so motion edges stay live."""
    import torch
    import torch.nn.functional as F
    g = frames[..., :3].float().mean(-1)                                     # [T,H,W]
    T = g.shape[0]
    d = torch.zeros_like(g)
    for k in (-2, -1, 1, 2):
        idx = torch.clamp(torch.arange(T) + k, 0, T - 1)
        d = torch.maximum(d, (g - g[idx]).abs())
    c = F.avg_pool2d(d[:, None], cell, cell, ceil_mode=True)[:, 0]           # [T,gh,gw]
    moving = (c >= thr).float()
    if grow:
        moving = F.max_pool2d(moving[:, None], 2 * grow + 1, 1, grow)[:, 0]
    return 1.0 - moving


class FastDeropeFreezeMask:
    """Per-frame MASK for H3 V2V Init: lock the gate's outer handle frames to pass 1 while pass 2 redraws the rest."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "the slowed clip (H3 Time Smear or Drawing Smear images)"}),
            "map": ("STRING", {"forceInput": True, "tooltip": "H3 Time Smear hold_map_used, or Drawing Smear drawing_map"}),
        }, "optional": {
            "segment_images": ("IMAGE", {"tooltip": "Motion Gate segment_images: enables freezing the STATIC background "
                               "(regions that don't move in pass 1) so pass 2 keeps its detail"}),
            "static_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.005,
                                 "tooltip": "grey change (0-1, per 16 px cell, within +-2 frames) below which a region "
                                            "counts as static. 0 = handles only"}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "mask"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Builds the regenerate/freeze mask for H3 V2V Init (set time_varying on): the outer handle frames the "
                   "Motion Gate marked are frozen to the pass-1 init, everything else is redrawn. Segment Splice then "
                   "crossfades only inside those frozen frames, so it never blends two different takes (no ghosting).")

    def mask(self, images, map, segment_images=None, static_threshold=0.03):
        import json
        import torch
        m = json.loads(map)
        if "src" in m:
            world = m["src"]
        else:
            world = [i for i, h in enumerate(m["holds"]) for _ in range(int(h))]
        freeze = m.get("freeze") or []
        t = images.shape[0]
        assert len(world) == t, (len(world), t)
        live = torch.tensor([0.0 if (w < len(freeze) and freeze[w]) else 1.0 for w in world])
        if segment_images is None or static_threshold <= 0:
            return (live[:, None, None].expand(t, 8, 8).contiguous(),)
        st = static_cells(segment_images.detach().cpu(), static_threshold)        # [world, gh, gw]
        st = st[torch.tensor(world).clamp(max=st.shape[0] - 1)]                    # onto the slowed clock
        return ((live[:, None, None] * (1.0 - st)).contiguous(),)
