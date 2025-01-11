__all__ = [
    'remove_long_silence',
    'remove_short_silence',
    'strip_silence',
]

from pydub import silence

from todd.patches.pydub import AudioSegment


def remove_long_silence(audio_segment: AudioSegment) -> AudioSegment:
    return sum(
        silence.split_on_silence(audio_segment, 1000, -50, 1000, 10),
        AudioSegment.empty(),
    )


def remove_short_silence(audio_segment: AudioSegment) -> AudioSegment:
    return sum(
        silence.split_on_silence(audio_segment, 100, -40, 1000, 10),
        AudioSegment.empty(),
    )


def strip_silence(audio_segment: AudioSegment) -> AudioSegment:
    start = silence.detect_leading_silence(audio_segment, -42)
    end = silence.detect_leading_silence(audio_segment.reverse(), -42, 1)
    assert start + end < len(audio_segment)
    return audio_segment[start:len(audio_segment) - end]
