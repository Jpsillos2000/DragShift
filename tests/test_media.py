"""Audio / video conversion (needs FFmpeg; skipped without it)."""
import subprocess

import pytest

from dragshift import convert, media
from dragshift.util import UserError, category

pytestmark = pytest.mark.skipif(not media.available(), reason="FFmpeg not available")


def ff(*args):
    subprocess.run([media.find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "Tone.wav"
    p.write_bytes(media.silent_wav(1.0))
    return p


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "Clip.mp4"
    ff("-f", "lavfi", "-i", "testsrc=size=161x121:rate=10:duration=1",
       "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4",  # mpeg4 allows odd sizes
       "-c:a", "aac", "-shortest", str(p))
    return p


@pytest.fixture
def silent_video(tmp_path):
    p = tmp_path / "Mute.mp4"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", str(p))
    return p


def test_categories(wav, video, tmp_path):
    assert category(wav) == "audio"
    assert category(video) == "video"
    (tmp_path / "x.mp3").write_bytes(b"")
    assert category(tmp_path / "x.mp3") == "audio"


def test_targets_exclude_source_format(wav, video):
    assert "WAV" not in convert.targets_for("audio", ".wav")
    assert "MP4" not in convert.targets_for("video", ".mp4")
    assert "MP3" in convert.targets_for("video", ".mp4")  # audio extraction


@pytest.mark.parametrize("target", convert.targets_for("audio", ".wav"))
def test_audio_targets(wav, target):
    out = convert.convert(wav, "audio", target)[0]
    info = media.probe(out)
    assert out.exists() and info and info.has_audio and not info.has_video
    assert info.duration == pytest.approx(1.0, abs=0.25)


@pytest.mark.parametrize("target", convert.targets_for("video", ".mp4"))
def test_video_targets(video, target):
    out = convert.convert(video, "video", target)[0]
    info = media.probe(out)
    assert out.exists() and info
    if target in media.VIDEO_TO_AUDIO:
        assert info.has_audio and not info.has_video
    else:
        assert info.has_video


def test_odd_dimensions_encode(video):
    out = convert.convert(video, "video", "MP4")[0]  # source is 161x121
    assert media.probe(out).has_video


def test_extract_audio_from_silent_video(silent_video):
    with pytest.raises(UserError, match="no audio"):
        convert.convert(silent_video, "video", "MP3")


def test_garbage_input(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(UserError):
        convert.convert(bad, "audio", "WAV")
    assert not list(tmp_path.glob(".~*"))  # no half-written leftovers


def test_never_overwrites(wav):
    a = convert.convert(wav, "audio", "FLAC")[0]
    b = convert.convert(wav, "audio", "FLAC")[0]
    assert a != b and a.exists() and b.exists()


def test_report_summary(video):
    from dragshift import report
    out = convert.convert(video, "video", "MKV")
    r = report.summarize("convert:MKV", [video], out)
    assert "MP4 → MKV" in r.title and "0:01" in r.detail
