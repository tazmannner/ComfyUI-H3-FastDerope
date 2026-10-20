"""Drawing De-rope: slow down and redraw only the DRAWINGS of a fast burst, not every frame.

The de-rope works because each fast pose gets its own static latent token (H3 draws static tokens crisply; it paints
blur on motion inside a token). But anime only shows a new drawing every 2-3 frames during action, so the in-between
world frames never need to exist. Drawing Smear keeps one frame per drawing slot inside the hot span and gives it
exactly one latent token (aligned to H3's (1,4,4,4,4) token grid); handles and shoulders are smeared as usual.
Drawing Recover takes each redrawn drawing and holds it over its 2- or 3-frame slot: the burst comes back crisp and
already on 2s/3s, from about half (2s) or a third (3s) of the de-rope's tokens."""
import json

import torch

LEGAL_STEP = 17


def _legal_ceil(n):
    k = max(2, -(-(n - 5) // LEGAL_STEP))
    return LEGAL_STEP * k + 5


def _is_quad_start(p):
    """first dilated frame of a 4-frame latent token (each 17 frames are tokens of (1,4,4,4,4) frames)"""
    return p % LEGAL_STEP in (1, 5, 9, 13)


def _next_quad_start(p):
    p += 1
    while not _is_quad_start(p):
        p += 1
    return p


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [0]):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i))
            s = None
    return out


def _split_gap(g, every):
    """slot lengths filling g world frames, as close to `every` as possible (anime holds of 2-3)"""
    if every == 1 or g <= max(every, 3):
        return [1] * g if every == 1 else [g]
    if every == 2:
        return [2] * (g // 2) if g % 2 == 0 else [2] * (g // 2 - 1) + [3]
    q, r = divmod(g, every)
    return [every] * q if r == 0 else ([every] * (q - 1) + [2, 2] if r == 1 else [every] * q + [2])


def key_scores(images, size=128):
    """Per-frame 'key pose' score: how much frame k is an extreme (direction reversal, impact) rather than a point on a
    straight path. e = (|k-(k-1)| + |k-(k+1)| - |(k+1)-(k-1)|) / (|k-(k-1)| + |k-(k+1)|), on the top-10% changed pixels
    so a small fast fist counts. Also returns the motion size (|k-(k-1)| + |k-(k+1)|) and the shot cuts."""
    x = images[..., :3].float().mean(-1)[:, None]
    T, _, H, W = x.shape
    sc = size / max(H, W)
    g = torch.nn.functional.interpolate(x, size=(max(8, int(H * sc)), max(8, int(W * sc))), mode="area")[:, 0]

    def d(a, b):
        v = (g[a] - g[b]).abs().flatten()
        return v.topk(max(1, v.numel() // 10)).values.mean().item()
    step = [0.0] + [d(k, k - 1) for k in range(1, T)]
    e, m = [0.0] * T, [0.0] * T
    for k in range(1, T - 1):
        a, b, c = step[k], step[k + 1], d(k + 1, k - 1)
        m[k] = a + b
        e[k] = max(0.0, (a + b - c) / (a + b + 1e-6))
    # shot cuts: one step far bigger than the steps around it. A drawing slot must never straddle a cut.
    # (a wholesale tone change: histogram distance; pixel steps can't tell a cut from a fast move on 2s)
    hist = [torch.histc(g[k], 32, 0.0, 1.0) for k in range(T)]
    hist = [h / h.sum() for h in hist]
    cuts = [k for k in range(1, T) if 0.5 * (hist[k] - hist[k - 1]).abs().sum().item() > 0.25]
    return e, m, cuts


def key_slots(h0, h1, every, e, m, cuts=(), min_e=0.35):
    """Slot starts for the hot run [h0, h1): every key pose (local max of e, real motion) starts a drawing, the gaps
    between keys are filled with holds of `every` (2-3), so the impact/extreme frames are always drawn."""
    mm = sorted(m[h0:h1])[len(m[h0:h1]) // 2] if h1 > h0 else 0.0
    cand = [k for k in range(h0 + 1, h1 - 1) if e[k] >= min_e and e[k] >= e[k - 1] and e[k] >= e[k + 1] and m[k] >= mm]
    keys = []
    for k in sorted(cand, key=lambda k: -e[k]):            # strongest first, at least 2 frames apart
        if all(abs(k - j) >= 2 for j in keys):
            keys.append(k)
    bounds = sorted(set([h0] + keys + [c for c in cuts if h0 < c < h1])) + [h1]
    starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        p = a
        for ln in _split_gap(b - a, every):
            starts.append(p)
            p += ln
    return starts, sorted(keys)


def plan_drawings(holds, hot, every, starts_for=None, key_tokens=1):
    """holds: the gate's per-world-frame holds of the segment; hot: 0/1 per frame; every: world frames per drawing.
    Returns (src, groups, slots): src = segment frame index for every dilated frame; groups[k] = (d0, d1) dilated range
    of world frame k's own group (drawings: the drawing's token); slots = [(world_first, world_end, drawing_frame)]."""
    n = len(holds)
    hot_runs = _runs(hot)
    src, groups, slots = [], [None] * n, []
    f = 0
    for (h0, h1) in hot_runs + [(n, n)]:
        while f < h0:                                         # normal smear outside the hot span
            g0 = len(src)
            src += [f] * int(holds[f])
            groups[f] = (g0, len(src))
            f += 1
        if h0 >= n:
            break
        # token-align the first drawing: the frame before it holds a little longer
        while not _is_quad_start(len(src)):
            if not src:
                src.append(h0)                                # clip starts hot: frame 0 is its own singleton token
                groups[h0] = (0, 1)
                continue
            src.append(src[-1])
            k = src[-1]
            groups[k] = (groups[k][0], len(src))
        st, keys = starts_for(h0, h1) if starts_for else (list(range(h0, h1, every)), [])
        for i, w0 in enumerate(st):
            w1 = st[i + 1] if i + 1 < len(st) else h1
            d0 = len(src)
            d1 = _next_quad_start(d0)   # one 4-frame token (+ the 17k singleton after it, if any: same drawing)
            for _ in range(key_tokens - 1 if w0 in keys else 0):
                d1 = _next_quad_start(d1)  # key poses (impacts, reversals) get more static tokens: pass 2 commits to them
            src += [w0] * (d1 - d0)
            for k in range(w0, w1):
                groups[k] = (d0, d1)
            slots.append((w0, w1, w0))
        f = h1
    # Pad to a legal length by stepping back through the last frames (ping-pong), not by repeating the last one: a long
    # run of one identical frame at the end makes H3 invent other content there (ghosts of an earlier shot), and the
    # last real frame shares its latent token with that run. The padding is never shown (no world frame maps to it).
    pad = _legal_ceil(len(src)) - len(src)
    order = list(dict.fromkeys(reversed(src)))       # distinct world frames, latest first
    bounce = order[1:] + order[::-1][1:] if len(order) > 1 else order
    src += [bounce[i % len(bounce)] for i in range(pad)]
    return src, groups, slots


def audio_holds(groups, start=0):
    """Per WORLD frame of the whole clip: how many slowed frames it occupies in pass 2. A 2s/3s drawing's token is
    shared by its 2-3 world frames, so its length is split between them in whole frames (5 over 2 -> 3, 2); calm gaps
    left out of pass 2 are 0. Frames before the segment are 0, so H3 Audio Smear skips the clip's audio up to there."""
    share, seen = {}, {}
    for g in groups:
        if g is not None:
            share[tuple(g)] = share.get(tuple(g), 0) + 1
    holds = [0] * int(start)
    for g in groups:
        if g is None:
            holds.append(0)
            continue
        k = tuple(g)
        i = seen.get(k, 0)
        seen[k] = i + 1
        q, r = divmod(g[1] - g[0], share[k])
        holds.append(int(q + (1 if i < r else 0)))
    return {"holds": holds, "world_len": len(holds)}


def _sharpness(x):
    g = x[..., :3].mean(-1)[:, None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=g.dtype)[None, None]
    return torch.nn.functional.conv2d(g, k).flatten(1).var(dim=1)


class FastDeropeDrawingSmear:
    """Slow a gated burst down by drawings: one static latent token per 2s/3s drawing inside the hot span."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Motion Gate segment_images"}),
            "hold_map": ("STRING", {"forceInput": True, "tooltip": "Motion Gate hold_map (carries the hot-frame mask)"}),
            "every": ("INT", {"default": 2, "min": 1, "max": 4,
                      "tooltip": "world frames per drawing inside the hot span: 2 = on twos, 3 = on threes. "
                                 "1 = a plain de-rope at hold 4 (every frame its own token)"}),
            "key_poses": ("BOOLEAN", {"default": True,
                          "tooltip": "start a drawing on every key pose (impact, direction reversal) instead of a fixed "
                                     "2s/3s phase, so extremes are never skipped"}),
        }, "optional": {
            "key_tokens": ("INT", {"default": 2, "min": 1, "max": 3,
                           "tooltip": "latent tokens per key-pose drawing (more = pass 2 holds the impact pose more firmly)"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "drawing_map", "length", "report", "audio_hold_map")
    OUTPUT_TOOLTIPS = (None, None, None, None,
                       "for MAINodes H3 Audio Smear (pass-1 audio -> VAEEncodeAudio -> H3 V2V Init audio_latent, audio "
                       "pinned): pass 2 then hears the real performance on its slowed clock instead of inventing speech "
                       "and moving the mouth to it")
    FUNCTION = "smear"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Drawing De-rope, step 1. Inside the gated hot span keeps one frame per drawing slot (every 2 or 3 "
                   "frames) and gives each its own token-aligned static latent token; handles and shoulders get the "
                   "gate's normal holds. Feed the output to VAE Encode -> H3 V2V Init as with H3 Time Smear, and wire "
                   "drawing_map to Drawing Recover.")

    def smear(self, images, hold_map, every, key_poses=True, key_tokens=2):
        hm = json.loads(hold_map)
        holds = hm["holds"]
        n = images.shape[0]
        assert len(holds) == n, f"hold map covers {len(holds)} frames, batch has {n}"
        hot = hm.get("hot") or [1 if h > 1 else 0 for h in holds]
        images = images.detach().cpu()
        keys_found = []
        starts_for = None
        if key_poses and every > 1:
            e, m, cuts = key_scores(images)

            def starts_for(h0, h1):
                st, ks = key_slots(h0, h1, every, e, m, cuts)
                keys_found.extend(ks)
                return st, ks
        src, groups, slots = plan_drawings(holds, hot, every, starts_for, key_tokens)
        dm = {"world_len": n, "src": src, "groups": groups, "slots": slots, "every": every, "freeze": hm.get("freeze") or [],
              "static": hm.get("static")}
        n_hot = int(sum(hot))
        report = (f"DrawingSmear: {n} segment frames ({n_hot} hot) -> {len(slots)} drawings on {every}s; "
                  f"{len(src)} frames to redraw (a hold-4 de-rope of the hot span alone would be {4 * n_hot})"
                  + (f"; key poses at {keys_found}" if keys_found else ""))
        print("[FastDerope] " + report)
        return (images[torch.tensor(src)], json.dumps(dm), len(src), report,
                json.dumps(audio_holds(groups, hm.get("start", 0))))


class FastDeropeDrawingRecover:
    """Back to real time: every frame of a drawing slot shows that drawing (the sharpest frame of its token)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "decoded pass-2 frames"}),
            "drawing_map": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recover"
    CATEGORY = "FastDerope"

    def recover(self, images, drawing_map):
        dm = json.loads(drawing_map)
        images = images.detach().cpu()
        assert images.shape[0] == len(dm["src"]), (images.shape[0], len(dm["src"]))
        in_slot = set(k for w0, w1, _ in dm["slots"] for k in range(w0, w1))
        best, pick = {}, []
        for k, (d0, d1) in enumerate(dm["groups"]):
            if k in in_slot:
                if (d0, d1) not in best:
                    best[(d0, d1)] = d0 + int(torch.argmax(_sharpness(images[d0:d1].float())).item())
                pick.append(best[(d0, d1)])
            else:
                pick.append(d0)                     # H3 Exact Recover convention: first frame of the hold group
        return (images[torch.tensor(pick)],)
