"""Retime the prompt's shot timestamps for pass 2.

Pass 2 redraws the gated segment on a SLOWED clock (each world frame held 1-4x), starting at the segment's first frame.
A multi-shot prompt's "At 00:04.500, the camera cuts to ..." then lands in the wrong place: in a 108-frame segment slowed
to ~9 s, the 4.5 s cut falls in the middle of the previous shot, and pass 2 blends the two shots (a later shot's tears
or background bleeding into this one). This node moves every timestamp onto the slowed clock and drops the shots that
start after the segment, so pass 2 is told what happens when, in its own time."""
import json
import re

TS = re.compile(r"At (\d+):(\d{2})\.(\d{3})")


def _dilated_starts(m):
    """per segment world frame: its first frame on the slowed clock"""
    if m.get("groups"):
        return [g[0] if g else 0 for g in m["groups"]]
    out, d = [], 0
    for h in m.get("holds", []):
        out.append(d)
        d += int(h)
    return out


def retime(prompt, seg_map, segment_start, fps=24.0):
    m = json.loads(seg_map)
    starts = _dilated_starts(m)
    n = len(starts)
    if n == 0:
        return prompt, "no segment"
    a, b = segment_start, segment_start + n - 1
    cut = prompt.find("overall_soundscape:")
    desc, rest = (prompt, "") if cut < 0 else (prompt[:cut], prompt[cut:])
    paras = [p for p in re.split(r"\n\s*\n", desc.strip()) if p.strip()]
    times = []
    for i, p in enumerate(paras):
        t = TS.search(p)
        times.append(int(t.group(1)) * 60 + int(t.group(2)) + int(t.group(3)) / 1000 if t else (0.0 if i == 0 else None))
    for i in range(1, len(times)):                      # a paragraph without a timestamp continues the one before it
        if times[i] is None:
            times[i] = times[i - 1]
    frames = [int(round(t * fps)) for t in times]
    keep, log = [], []
    last_before = max([i for i in range(1, len(paras)) if frames[i] <= a] or [0])
    for i, p in enumerate(paras):
        if i > 0 and frames[i] > b:
            log.append(f"drop para {i + 1} (starts f{frames[i]} after the segment f{a}-f{b})")
            continue
        if 0 < i < last_before:
            log.append(f"drop para {i + 1} (ends before the segment)")
            continue
        if i > 0 and TS.search(p):
            d = starts[max(0, frames[i] - a)] / fps
            new = f"At {int(d // 60):02d}:{int(d % 60):02d}.{int(round((d % 1) * 1000)) % 1000:03d}"
            log.append(f"para {i + 1}: f{frames[i]} -> {new[3:]}")
            p = TS.sub(new, p, count=1)
        keep.append(p)
    out = "\n\n".join(keep) + ("\n\n" + rest if rest else "")
    return out, "; ".join(log) or "no timestamps"


class FastDeropeRetimePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "tooltip": "the pass-1 prompt"}),
            "seg_map": ("STRING", {"forceInput": True, "tooltip": "Drawing Smear drawing_map (or Time Smear hold_map_used)"}),
            "segment_start": ("INT", {"forceInput": True, "tooltip": "Motion Gate segment_start"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "run"
    CATEGORY = "FastDerope"
    DESCRIPTION = ("Pass 2 redraws the burst on a slowed clock, so the prompt's 'At 00:0X.XXX' shot timestamps must move "
                   "too. Retimes them onto pass 2's clock and drops shots that start after the segment. Wire its prompt "
                   "into pass 2's MiniMax H3 Image to Video.")

    def run(self, prompt, seg_map, segment_start, fps=24.0):
        out, rep = retime(prompt, seg_map, segment_start, fps)
        print("[FastDerope] RetimePrompt: " + rep)
        return (out, rep)
