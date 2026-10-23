"""Audio and video conversion through FFmpeg. Same contract as convert.py: every function takes a
source path and returns a list of output paths written next to the source."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .util import UserError, sibling

# Anything FFmpeg can read counts as input; the lists below only decide the menu category.
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".m4b", ".ogg", ".oga", ".opus", ".wma",
              ".aiff", ".aif", ".aifc", ".ac3", ".eac3", ".dts", ".amr", ".ape", ".wv", ".mka",
              ".mp2", ".mp1", ".au", ".snd", ".caf", ".spx", ".tta", ".mpc", ".w64", ".voc", ".gsm", ".ra",
              ".m4r", ".m4p", ".aax", ".oma", ".tak", ".dsf", ".dff", ".mlp", ".thd", ".sbc", ".adts"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg",
              ".mpe", ".3gp", ".3g2", ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".asf", ".f4v",
              ".rm", ".rmvb", ".divx", ".xvid", ".mxf", ".dv", ".qt", ".yuv", ".y4m", ".mpeg4", ".mp4v",
              ".m1v", ".m2v", ".mpv", ".mpv2", ".mjpeg", ".mjpg", ".h264", ".264", ".h265", ".265",
              ".hevc", ".avc", ".ogm", ".amv", ".wtv", ".dvr-ms", ".trp", ".tp", ".m2t", ".evo", ".gxf",
              ".nut", ".nsv", ".roq", ".bik", ".smk", ".swf", ".vro", ".mod", ".tod", ".mk3d", ".ivf"}
# containers that can hold either: look inside to decide
AMBIGUOUS_EXTS = {".mp4", ".mkv", ".webm", ".ogg", ".3gp", ".3g2", ".asf", ".ts", ".mov", ".m4a",
                  ".mpeg4", ".mp4v", ".ogm", ".nut", ".mod"}
# never worth probing: these are other DragShift categories or plainly not media
_NOT_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".tif", ".tiff", ".avif", ".bmp",
                   ".ico", ".pdf", ".docx", ".txt", ".md", ".csv", ".log", ".json", ".xml", ".html", ".htm",
                   ".exe", ".dll", ".zip", ".lnk", ".py", ".js", ".iso", ".dmg", ".msi", ".app"}

# target -> (file extension, FFmpeg muxer, codec arguments)
_X264 = ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:a", "aac", "-b:a", "160k"]
AUDIO_SPECS: dict[str, tuple[str, str, list[str]]] = {
    "MP3":  (".mp3",  "mp3",    ["-c:a", "libmp3lame", "-q:a", "2"]),
    "WAV":  (".wav",  "wav",    ["-c:a", "pcm_s16le"]),
    "FLAC": (".flac", "flac",   ["-c:a", "flac"]),
    "M4A":  (".m4a",  "mp4",    ["-c:a", "aac", "-b:a", "192k"]),
    "AAC":  (".aac",  "adts",   ["-c:a", "aac", "-b:a", "192k"]),
    "OGG":  (".ogg",  "ogg",    ["-c:a", "libvorbis", "-q:a", "5"]),
    "OPUS": (".opus", "opus",   ["-c:a", "libopus", "-b:a", "128k"]),
    "WMA":  (".wma",  "asf",    ["-c:a", "wmav2", "-b:a", "192k"]),
    "AIFF": (".aiff", "aiff",   ["-c:a", "pcm_s16be"]),
}
VIDEO_SPECS: dict[str, tuple[str, str, list[str]]] = {
    "MP4":  (".mp4",  "mp4",      _X264 + ["-movflags", "+faststart"]),
    "MKV":  (".mkv",  "matroska", _X264),
    "WEBM": (".webm", "webm",     ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-row-mt", "1",
                                   "-deadline", "good", "-cpu-used", "4", "-pix_fmt", "yuv420p",
                                   "-c:a", "libopus", "-b:a", "128k"]),
    "MOV":  (".mov",  "mov",      _X264),
    "AVI":  (".avi",  "avi",      ["-c:v", "mpeg4", "-q:v", "4", "-c:a", "libmp3lame", "-q:a", "3"]),
    "GIF":  (".gif",  "gif",      ["-an", "-loop", "0", "-vf",
                                   "fps=15,scale='min(480,iw)':-2:flags=lanczos,split[a][b];"
                                   "[a]palettegen[p];[b][p]paletteuse"]),
}
# video -> audio: pull the soundtrack out with the same settings as audio targets
VIDEO_TO_AUDIO = ["MP3", "M4A", "WAV", "FLAC"]
AUDIO_TARGETS = list(AUDIO_SPECS)
VIDEO_TARGETS = list(VIDEO_SPECS) + VIDEO_TO_AUDIO

_ENCODER = {"MP3": "libmp3lame", "WAV": "pcm_s16le", "FLAC": "flac", "M4A": "aac", "AAC": "aac",
            "OGG": "libvorbis", "OPUS": "libopus", "WMA": "wmav2", "AIFF": "pcm_s16be"}
_VIDEO_ENCODER = {"MP4": "libx264", "MKV": "libx264", "MOV": "libx264", "WEBM": "libvpx-vp9",
                  "AVI": "mpeg4", "GIF": "gif"}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash from the windowed exe


# ---------------------------------------------------------------- locating FFmpeg
@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    """DRAGSHIFT_FFMPEG, then the copy shipped with the app (imageio-ffmpeg), then PATH."""
    env = os.environ.get("DRAGSHIFT_FFMPEG")
    if env and os.path.isfile(env):
        return env
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def available() -> bool:
    return find_ffmpeg() is not None


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    exe = find_ffmpeg()
    if not exe:
        raise UserError("FFmpeg isn't installed. Install it (or `pip install imageio-ffmpeg`) "
                        "to convert audio and video.")
    return subprocess.run([exe, "-hide_banner", "-nostdin", *args], capture_output=True,
                          timeout=timeout, creationflags=_NO_WINDOW)


@lru_cache(maxsize=1)
def _encoders() -> frozenset[str]:
    try:
        text = _run(["-encoders"], timeout=20).stdout.decode("utf-8", "replace")
    except Exception:
        return frozenset()
    return frozenset(m.group(1) for m in re.finditer(r"^\s[VAS][\w.]{5}\s+(\S+)", text, re.M))


# ---------------------------------------------------------------- probing
class Probe:
    def __init__(self, has_video: bool, has_audio: bool, duration: float | None):
        self.has_video, self.has_audio, self.duration = has_video, has_audio, duration


@lru_cache(maxsize=64)
def _probe(path: str, mtime: float) -> Probe | None:
    try:
        err = _run(["-i", path], timeout=20).stderr.decode("utf-8", "replace")
    except Exception:
        return None
    streams = re.findall(r"Stream #\S+?: (Video|Audio):(.*)", err)
    # cover art is "Video" too, but it's a still image, not a video track
    video = any(k == "Video" and "attached pic" not in rest for k, rest in streams)
    audio = any(k == "Audio" for k, _ in streams)
    dur = None
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if not streams and "Duration" not in err:
        return None
    return Probe(video, audio, dur)


def probe(path: str | Path) -> Probe | None:
    p = Path(path)
    try:
        return _probe(str(p), p.stat().st_mtime)
    except OSError:
        return None


def duration(path: str | Path) -> float | None:
    info = probe(path)
    return info.duration if info else None


def kind(path: str | Path) -> str | None:
    """'audio', 'video' or None. Extension first; look inside only for containers that can be both."""
    ext = Path(path).suffix.lower()
    if ext in AMBIGUOUS_EXTS and available():
        info = probe(path)
        if info:
            return "video" if info.has_video else "audio" if info.has_audio else None
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext not in _NOT_MEDIA_EXTS and available():
        # unknown extension (.mpeg4 renamed, .bin, no extension…): let FFmpeg look inside
        try:
            if Path(path).is_file() and Path(path).stat().st_size > 1024:
                info = probe(path)
                if info:
                    return "video" if info.has_video else "audio" if info.has_audio else None
        except OSError:
            pass
    return None


def human_duration(sec: float | None) -> str:
    if not sec:
        return ""
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------- converting
def targets_for(cat: str, src_ext: str) -> list[str]:
    """Targets FFmpeg can really produce on this machine, minus the source's own format."""
    if not available():
        return []
    have = _encoders()
    cur = src_ext.lower().lstrip(".").upper()
    cur = {"OGA": "OGG", "AIF": "AIFF", "AIFC": "AIFF", "M4B": "M4A"}.get(cur, cur)
    out = []
    for t in (AUDIO_TARGETS if cat == "audio" else VIDEO_TARGETS):
        need = _ENCODER.get(t) if (cat == "audio" or t in VIDEO_TO_AUDIO) else _VIDEO_ENCODER[t]
        if t != cur and (not have or need in have):
            out.append(t)
    return out


def _fail_message(stderr: str) -> str:
    low = stderr.lower()
    if "does not contain any stream" in low or "matches no streams" in low:
        return "This file has no audio track to extract."
    if "invalid data found" in low or "moov atom not found" in low:
        return "This file is damaged or isn't a media file."
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    return lines[-1][:200] if lines else "FFmpeg couldn't convert this file."


def convert_media(src: str | Path, cat: str, target: str) -> list[Path]:
    target = target.upper()
    src = Path(src)
    to_audio = target in AUDIO_SPECS and (cat == "audio" or target in VIDEO_TO_AUDIO)
    specs = AUDIO_SPECS if to_audio else VIDEO_SPECS
    if target not in specs:
        raise ValueError(f"Can't convert {cat} to {target}")
    ext, muxer, codec_args = specs[target]
    info = probe(src)
    if to_audio and info and not info.has_audio:
        raise UserError("This file has no audio track to extract.")
    out = sibling(src, ext=ext)
    tmp = out.with_name(f".~{out.name}.part")

    maps = ["-map", "0:a:0", "-map_metadata", "0"] if to_audio else \
           ["-map", "0:v:0", "-map", "0:a?", "-sn"]
    if to_audio:
        maps.insert(0, "-vn")
    try:
        # MKV takes almost any stream: try a lossless remux first, re-encode only if that fails
        attempts = [["-c", "copy"]] if target == "MKV" and cat == "video" else []
        attempts.append(codec_args)
        if not to_audio:
            attempts.append(codec_args + ["-ac", "2"])  # e.g. 5.1 sound that libopus/mp3 can't take as-is
        last = ""
        for codec in attempts:
            r = _run(["-y", "-i", str(src), *maps, *codec, "-f", muxer, str(tmp)])
            ok = r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0
            if ok and not to_audio and target != "GIF" and info and info.has_audio:
                got = _probe(str(tmp), tmp.stat().st_mtime)
                ok = bool(got and got.has_audio)  # never hand back a silent copy of a video with sound
                if not ok:
                    last = "The sound track was lost during conversion."
            if ok:
                os.replace(tmp, out)
                return [out]
            last = r.stderr.decode("utf-8", "replace") or last
        raise UserError(_fail_message(last))
    except subprocess.TimeoutExpired:  # pragma: no cover
        raise UserError("FFmpeg took too long.")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def silent_wav(seconds: float = 0.3, rate: int = 8000) -> bytes:
    """A tiny valid WAV (a quiet 440 Hz tone), for self-tests."""
    import io
    import math
    import struct
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / rate)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()
