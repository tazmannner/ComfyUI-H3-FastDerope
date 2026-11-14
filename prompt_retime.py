"""Retime the prompt's shot timestamps for pass 2.

Pass 2 redraws the gated segment on a SLOWED clock (each world frame held 1-4x), starting at the segment's first frame.
A multi-shot prompt's "At 00:04.500, the camera cuts to ..." then lands in the wrong place: in a 108-frame segment slowed
to ~9 s, the 4.5 s cut falls in the middle of the previous shot, and pass 2 blends the two shots (a later shot's tears
or background bleeding into this one). This node moves every timestamp onto the slowed clock and drops the shots that
start after the segment, so pass 2 is told what happens when, in its own time.

Beats timed in seconds ("OPENING - 0-1.5 SECONDS:", "ACTION (1.5-3.5s):", "0:01.5-0:03.5:") are retimed the same way:
a beat runs until the next beat, shot line or header, keeps its stated end, and is clipped to the segment."""
import json
import re

TS = re.compile(r"At (\d+):(\d{2})\.(\d{3})")
SHOT = re.compile(r"^\s*(\[Shot \d+\]\s*)?(At \d+:\d{2}\.\d{3})?")
# a beat line: optional label, a time or time range in seconds or m:ss(.s), optional unit, then ":" or the end of the line
T = r"\d+:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?"
UNIT = r"(?:s|secs?|seconds?)\b"
BEAT = re.compile(rf"^\s*(?P<label>[^\d\n:]*?)[\s(\[\-–—:]*(?P<a>{T})(?:\s*[-–—]\s*(?P<b>{T}))?\s*(?P<unit>{UNIT})?\s*[)\]]?\s*(?P<colon>:)?",
                  re.I)
HEADER = re.compile(r"^\s*(?:[A-Z][A-Z0-9 /&'_\-]{1,40}|[a-z_]+):(\s|$)")   # "CAMERA:", "overall_soundscape:"
TIMEISH = re.compile(rf"\b\d+(?:\.\d+)?\s*{UNIT}|\b\d+:\d{{2}}(?:\.\d+)?\b", re.I)


def _dilated_starts(m):
    """per segment world frame: its first frame on the slowed clock, plus the slowed clock's length"""
    if m.get("groups"):
        starts = [g[0] if g else 0 for g in m["groups"]]
        return starts, max([max(g) for g in m["groups"] if g] or [0]) + 1
    out, d = [], 0
    for h in m.get("holds", []):
        out.append(d)
        d += int(h)
    return out, d


def _shot_start(line):
    """a line opens a shot when it starts with "[Shot N]" and/or an "At mm:ss.mmm" timestamp"""
    m = SHOT.match(line)
    return bool(m and (m.group(1) or m.group(2)))


def _beat(line):
    """a beat line: "LABEL - 0-1.5 SECONDS:", "1.5-3.5s", "ACTION (2 s):", "0:01.5-0:03.5:". The time ends the line or
    is followed by a colon, the label is at most 4 words; a single time needs a unit and a colon, a range needs a unit,
    a colon or the m:ss form (so prose like "she walks 2-3 s away" or "2-3 people" is not a beat)"""
    if _shot_start(line):
        return None
    m = BEAT.match(line)
    if not m or len(m.group("label").split()) > 4 or not (m.group("colon") or not line[m.end():].strip()):
        return None
    clock = ":" in m.group("a") or ":" in (m.group("b") or "")
    if m.group("b"):
        return m if (m.group("unit") or m.group("colon") or clock) else None
    return m if (m.group("unit") and m.group("colon")) else None


def _secs(t):
    return int(t.group(1)) * 60 + int(t.group(2)) + int(t.group(3)) / 1000


def _beat_secs(s):
    if ":" in s:
        mm, ss = s.split(":")
        return int(mm) * 60 + float(ss)
    return float(s)


def _fmt_beat(sec, like):
    """a retimed beat time written like the original (m:ss.s or plain seconds)"""
    if ":" in like:
        return f"{int(sec // 60)}:{sec % 60:04.1f}" if "." in like else f"{int(sec // 60)}:{int(round(sec % 60)):02d}"
    return f"{sec:.2f}".rstrip("0").rstrip(".")


def retime(prompt, seg_map, segment_start, fps=24.0):
    """Shots are the lines that open with "[Shot N]" or "At mm:ss.mmm"; each runs until the next shot line or a blank
    line. Beats are lines timed in seconds; each runs (across blank lines) until the next beat, shot line or header
    ("CAMERA:", "overall_soundscape:"). Everything else (definitions, summary, setting, style, untimed paragraphs) is
    kept as it is. Shots/beats that start after the segment or end before it are dropped; a beat that straddles an edge
    is kept and clipped to the segment; every time left is moved onto pass 2's slowed clock. A prompt without
    timestamps comes back unchanged."""
    m = json.loads(seg_map)
    starts, total = _dilated_starts(m)
    n = len(starts)
    if n == 0:
        return prompt, "no segment"
    cut = prompt.find("overall_soundscape:")
    desc, rest = (prompt, "") if cut < 0 else (prompt[:cut], prompt[cut:])
    lines = desc.split("\n")
    beats = {i: b for i, line in enumerate(lines) if (b := _beat(line))}
    if not TS.search(prompt) and not beats:
        warn = sorted({t.group(0) for t in TIMEISH.finditer(desc)})
        return prompt, "no timestamps" + (f"; WARNING time-like text not understood: {', '.join(warn)}" if warn else "")
    a, b = segment_start, segment_start + n - 1

    def clock(f):                            # world frame -> seconds on pass 2's clock (end-exclusive past the end)
        return (starts[min(n - 1, max(0, f - a))] if f - a < n else total) / fps

    shots, cur = [], None                    # shots: [first line, last line, start frame]
    units = []                               # beats: [first line, last line, start frame, end frame or None]
    for i, line in enumerate(lines):
        if _shot_start(line):
            t = TS.search(line)
            f = int(round(_secs(t) * fps)) if t else (shots[-1][2] if shots else 0)
            cur = [i, i, f]
            shots.append(cur)
        elif i in beats:
            bt = beats[i]
            end = int(round(_beat_secs(bt.group("b")) * fps)) if bt.group("b") else None
            cur = [i, i, int(round(_beat_secs(bt.group("a")) * fps)), end]
            units.append(cur)
        elif cur is not None and len(cur) == 4 and not HEADER.match(line):
            cur[1] = i                       # a beat runs across blank lines until the next marker or header
        elif cur is not None and len(cur) == 3 and line.strip():
            cur[1] = i                       # the shot continues on the next line of its paragraph
        else:
            cur = None                       # a blank line ends a shot; a header ends a beat; what follows is kept
    log, drop, edit = [], set(), {}
    last_before = max([k for k, s in enumerate(shots) if s[2] <= a] or [0])
    for k, (l0, l1, f) in enumerate(shots):
        why = (f"starts f{f} after the segment f{a}-f{b}" if f > b and k > 0
               else "ends before the segment" if k < last_before else None)
        if why:
            log.append(f"drop shot {k + 1} ({why})")
            drop.update(range(l0, l1 + 1))
    for k, u in enumerate(units):
        l0, l1, f0, f1 = u
        bt = beats[l0]
        name = (bt.group("label").strip(" -–—(") or f"beat {k + 1}")[:30]
        # no stated end: it runs to the next beat (or to the end of the clip)
        end = f1 if f1 is not None else next((v[2] for v in units[k + 1:] if v[2] > f0), None)
        span = f"{bt.group('a')}{'-' + bt.group('b') if bt.group('b') else ''}"
        if f0 > b or (end is not None and end <= a):
            log.append(f"drop {name} {span} ({'starts after' if f0 > b else 'ends before'} the segment f{a}-f{b})")
            drop.update(range(l0, l1 + 1))
            continue
        na = _fmt_beat(clock(max(f0, a)), bt.group("a"))
        if bt.group("b"):
            nb = _fmt_beat(clock(min(f1, b + 1)), bt.group("b"))
            new = bt.group(0)[:bt.start("a") - bt.start(0)] + na + bt.group(0)[bt.end("a") - bt.start(0):bt.start("b") - bt.start(0)] + nb + bt.group(0)[bt.end("b") - bt.start(0):]
            clip = " (clipped at the segment " + ("start" if f0 < a else "") + (" and " if f0 < a and f1 > b + 1 else "") + ("end" if f1 > b + 1 else "") + ")" if (f0 < a or f1 > b + 1) else ""
            log.append(f"{name} {span} -> {na}-{nb}{clip}")
        else:
            new = bt.group(0)[:bt.start("a") - bt.start(0)] + na + bt.group(0)[bt.end("a") - bt.start(0):]
            log.append(f"{name} {span} -> {na}")
        edit[l0] = new + lines[l0][bt.end(0):]

    def move(t):
        f = int(round(_secs(t) * fps))
        d = starts[min(n - 1, max(0, f - a))] / fps
        new = f"At {int(d // 60):02d}:{int(d % 60):02d}.{int(round((d % 1) * 1000)) % 1000:03d}"
        log.append(f"f{f} -> {new[3:]}")
        return new
    kept = [TS.sub(move, edit.get(i, line)) for i, line in enumerate(lines) if i not in drop]
    # time-like text that is neither an "At" timestamp nor a beat: report it, since it passes through on the old clock
    warn = sorted({t.group(0) for i, line in enumerate(lines) if i not in beats and i not in drop
                   for t in TIMEISH.finditer(TS.sub("", line))})
    if warn:
        log.append("WARNING time-like text not understood (left on the original clock): " + ", ".join(warn))
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
    DESCRIPTION = ("Pass 2 redraws the burst on a slowed clock, so the prompt's 'At 00:0X.XXX' shot timestamps and its "
                   "beats timed in seconds ('OPENING - 0-1.5 SECONDS:') must move too. Retimes them onto pass 2's clock, "
                   "drops shots and beats outside the segment, clips beats that straddle it, and lists what it did in "
                   "'report' (wire it to a Preview Any node to check). Wire its prompt into pass 2's MiniMax H3 Image to "
                   "Video.")

    def run(self, prompt, seg_map, segment_start, fps=24.0):
        out, rep = retime(prompt, seg_map, segment_start, fps)
        print("[FastDerope] RetimePrompt: " + rep)
        return (out, rep)
