import re
import wave

import numpy

from .synth import generate_silence, generate_sin_wave

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_VOLUME = 1.0
DEFAULT_TONE = 800

DIT = object()
DAH = object()
SYMBOL_SPACE = object()
LETTER_SPACE = object()
WORD_SPACE = object()

MORSE_TABLE = {
    'a': (DIT, DAH),
    'b': (DAH, DIT, DIT, DIT),
    'c': (DAH, DIT, DAH, DIT),
    'd': (DAH, DIT, DIT),
    'e': (DIT, ),
    'f': (DIT, DIT, DAH, DIT),
    'g': (DAH, DAH, DIT),
    'h': (DIT, DIT, DIT, DIT),
    'i': (DIT, DIT),
    'j': (DIT, DAH, DAH, DAH),
    'k': (DAH, DIT, DAH),
    'l': (DIT, DAH, DIT, DIT),
    'm': (DAH, DAH),
    'n': (DAH, DIT),
    'o': (DAH, DAH, DAH),
    'p': (DIT, DAH, DAH, DIT),
    'q': (DAH, DAH, DIT, DAH),
    'r': (DIT, DAH, DIT),
    's': (DIT, DIT, DIT),
    't': (DAH, ),
    'u': (DIT, DIT, DAH),
    'v': (DIT, DIT, DIT, DAH),
    'w': (DIT, DAH, DAH),
    'x': (DAH, DIT, DIT, DAH),
    'y': (DAH, DIT, DAH, DAH),
    'z': (DAH, DAH, DIT, DIT),
    '0': (DAH, DAH, DAH, DAH, DAH),
    '1': (DIT, DAH, DAH, DAH, DAH),
    '2': (DIT, DIT, DAH, DAH, DAH),
    '3': (DIT, DIT, DIT, DAH, DAH),
    '4': (DIT, DIT, DIT, DIT, DAH),
    '5': (DIT, DIT, DIT, DIT, DIT),
    '6': (DAH, DIT, DIT, DIT, DIT),
    '7': (DAH, DAH, DIT, DIT, DIT),
    '8': (DAH, DAH, DAH, DIT, DIT),
    '9': (DAH, DAH, DAH, DAH, DIT),
    '.': (DIT, DAH, DIT, DAH, DIT, DAH),
    ',': (DAH, DAH, DIT, DIT, DAH, DAH),
    '/': (DAH, DIT, DIT, DAH, DIT),
    '?': (DIT, DIT, DAH, DAH, DIT, DIT),
    '=': (DAH, DIT, DIT, DIT, DAH),
    "'": (DIT, DAH, DAH, DAH, DAH, DIT),
    '!': (DAH, DIT, DAH, DIT, DAH, DAH),
    '(': (DAH, DIT, DAH, DAH, DIT),
    ')': (DAH, DIT, DAH, DAH, DIT, DAH),
    '&': (DIT, DAH, DIT, DIT, DIT),
    ':': (DAH, DAH, DAH, DIT, DIT, DIT),
    ';': (DAH, DIT, DAH, DIT, DAH, DIT),
    '+': (DIT, DAH, DIT, DAH, DIT),
    '-': (DAH, DIT, DIT, DIT, DIT, DAH),
    '_': (DIT, DIT, DAH, DAH, DIT, DAH),
    '"': (DIT, DAH, DIT, DIT, DAH, DIT),
    '$': (DIT, DIT, DIT, DAH, DIT, DIT, DAH),
}


def generate(
    text: str, wpm: int, tone: int = DEFAULT_TONE,
    volume: float = DEFAULT_VOLUME, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> numpy.array:
    text = normalize_text(text)
    samples = list(_generate_samples(text, wpm, tone, volume, sample_rate))
    return numpy.concatenate(samples)


def stream_wave(
    fp: wave.Wave_write, text: str, wpm: int, tone: int = DEFAULT_TONE,
    volume: float = DEFAULT_VOLUME, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> None:
    text = normalize_text(text)
    for sample in _generate_samples(text, wpm, tone, volume, sample_rate):
        fp.writeframes(sample)


def output_wave(
    file: str, text: str, wpm: int, tone: int = DEFAULT_TONE,
    volume: float = DEFAULT_VOLUME, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> None:
    text = normalize_text(text)
    with wave.open(file, "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(sample_rate)
        for sample in _generate_samples(text, wpm, tone, volume, sample_rate):
            fp.writeframes(sample)


def _generate_samples(
    text: str, wpm: int, tone: int = DEFAULT_TONE,
    volume: float = DEFAULT_VOLUME, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> numpy.array:
    dit_duration = 1.2 / wpm
    dah_duration = dit_duration * 3
    symbol_space_duration = dit_duration
    letter_space_duration = (dit_duration * 3) - symbol_space_duration
    word_space_duration = (dit_duration * 7) - letter_space_duration

    audio_params = {
        'attack': dit_duration / 10,
        'release': dit_duration / 10,
        'volume': volume,
        'sample_rate': sample_rate
    }

    samples = {
        DIT: generate_sin_wave(tone, dit_duration, **audio_params),
        DAH: generate_sin_wave(tone, dah_duration, **audio_params),
        SYMBOL_SPACE: generate_silence(symbol_space_duration),
        LETTER_SPACE: generate_silence(letter_space_duration),
        WORD_SPACE: generate_silence(word_space_duration),
    }

    def _encode_letter(letter: str):
        if letter == ' ':
            yield samples[WORD_SPACE]
            return

        symbols = MORSE_TABLE.get(letter)
        if not symbols:
            raise ValueError('Unsupported symbol: {}'.format(repr(letter)))

        for symbol in symbols:
            yield samples[symbol]
            yield samples[SYMBOL_SPACE]
        yield samples[LETTER_SPACE]

    for letter in normalize_text(text):
        yield from _encode_letter(letter)


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return text