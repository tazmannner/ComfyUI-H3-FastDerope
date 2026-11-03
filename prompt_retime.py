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


SHOT = re.compile(r"^\s*(\[Shot \d+\]\s*)?(At \d+:\d{2}\.\d{3})?")


def _shot_start(line):
    """a line opens a shot when it starts with "[Shot N]" and/or an "At mm:ss.mmm" timestamp"""
    m = SHOT.match(line)
    return bool(m and (m.group(1) or m.group(2)))


def _secs(t):
    return int(t.group(1)) * 60 + int(t.group(2)) + int(t.group(3)) / 1000


def retime(prompt, seg_map, segment_start, fps=24.0):
    """Shots are the lines that open with "[Shot N]" or "At mm:ss.mmm"; each runs until the next shot line or a blank
    line. Everything else (definitions, summary, setting, style, untimed paragraphs) is kept as it is. Shots that start
    after the segment or end before it are dropped; every timestamp left is moved onto pass 2's slowed clock. A prompt
    without timestamps comes back unchanged."""
    m = json.loads(seg_map)
    starts = _dilated_starts(m)
    n = len(starts)
    if n == 0:
        return prompt, "no segment"
    if not TS.search(prompt):
        return prompt, "no timestamps"
    a, b = segment_start, segment_start + n - 1
    cut = prompt.find("overall_soundscape:")
    desc, rest = (prompt, "") if cut < 0 else (prompt[:cut], prompt[cut:])
    lines = desc.split("\n")
    shots, cur = [], None                    # shots: [first line, last line, start frame]
    for i, line in enumerate(lines):
        if _shot_start(line):
            t = TS.search(line)
            f = int(round(_secs(t) * fps)) if t else (shots[-1][2] if shots else 0)
            cur = [i, i, f]
            shots.append(cur)
        elif cur is not None and line.strip():
            cur[1] = i                       # the shot continues on the next line of its paragraph
        else:
            cur = None                       # a blank line ends the shot; what follows is kept
    log, drop = [], set()
    last_before = max([k for k, s in enumerate(shots) if s[2] <= a] or [0])
    for k, (l0, l1, f) in enumerate(shots):
        why = (f"starts f{f} after the segment f{a}-f{b}" if f > b and k > 0
               else "ends before the segment" if k < last_before else None)
        if why:
            log.append(f"drop shot {k + 1} ({why})")
            drop.update(range(l0, l1 + 1))

    def move(t):
        f = int(round(_secs(t) * fps))
        d = starts[min(n - 1, max(0, f - a))] / fps
        new = f"At {int(d // 60):02d}:{int(d % 60):02d}.{int(round((d % 1) * 1000)) % 1000:03d}"
        log.append(f"f{f} -> {new[3:]}")
        return new
    kept = [TS.sub(move, line) for i, line in enumerate(lines) if i not in drop]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).rstrip() + ("\n\n" + rest if rest else "")
    return out, "; ".join(log) or "timestamps kept"


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
