# pycw

![Project Version](https://img.shields.io/pypi/v/pycw) ![Python Version](https://img.shields.io/pypi/pyversions/pycw)

Python Morse Code (CW) audio generator **and neural decoder**.

- **Generate** Morse audio into WAV files (`output_wave` / `generate` / streaming).
- **Decode** Morse audio back to text with a tiny neural model
  (`decode_wav` / `decode_samples` / `decode_bytes`, plus `pycw -d`).

The decoder is **numpy-only** — no PyTorch at runtime. The weights
(`pycw/decoder/model.bin`, ~374 KB) ship inside the package and come from the
[cw-train](https://git-direct.ghink.net/bigsk/cw-train) repository, which is
the single source of truth for the model (pycw, gocw and torch all reproduce
the same golden vectors bit-for-bit).

## Usage

### Generate CW audio

```bash
pycw -t "CQ CQ DE BD8CMN" -s 20 -n 700 -o intro.wav            # 20 wpm, 700 Hz
pycw -i message.txt -o message.wav -s 20 -n 600 -r 48000       # from a file
echo "sos" | pycw -o sos.wav                                   # from stdin
```

Or in code:

```python
import pycw
pycw.output_wave("Intro.wav", "CQ CQ CQ DE BD8CMN PSE K", 20)   # -> Intro.wav
```

### Decode CW audio

```bash
pycw -d received.wav                # auto-detects tone; any sample rate
pycw -d received.wav --decoder-tone 700
```

Or in code:

```python
import pycw

pycw.decode_wav("received.wav")             # file -> text
pycw.decode_samples(samples, 16000)         # float32 mono in [-1, 1] -> text
pycw.decode_bytes(pcm_bytes, 16000)         # raw int16 PCM bytes -> text

dec = pycw.Decoder()                        # reuse one instance for live use
dec.decode(samples, 16000)
```

### Round trip

```bash
pycw -t "cq cq de bd8cmn" -s 20 -n 700 -r 16000 -o t.wav
pycw -d t.wav                               # prints: cq cq de bd8cmn
```

### Tests

```bash
pip install numpy pytest
pytest tests/ -q                            # or: python tests/test_decoder.py
```

Covers: golden parity vs the torch reference (|dp| < 5e-4 on 6 clips),
full-pipeline text equality, WAV round trip, generate-then-decode round trip,
auto tone detection, and a guard that the decoder never imports torch.

## CLI reference

```
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input text file (defaults to stdin)
  -t TEXT, --text TEXT  Input text. Overrides --input.
  -s SPEED, --speed SPEED
                        Speed, in words per minute (default: 12)
  -n TONE, --tone TONE  Tone frequency, in Hz (default: 800)
  -v VOLUME, --volume VOLUME
                        Volume (default: 1.0)
  -r SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        Sample rate (default: 44100)
  -o OUTPUT, --output OUTPUT
                        Name of the output file
  -d DECODE, --decode DECODE
                        Decode a WAV file into Morse text (neural decoder)
  --decoder-tone DECODER_TONE
                        Fixed tone for decoding (default: auto-detect)
```

## Notes

- Robust to background noise, fading, hand-key jitter and QRM: CER ~0.02–0.05
  at -8…+12 dB SNR (full measurement table in the cw-train README).
- `pip install pycw[train]` adds torch **only** if you want to retrain models
  (training lives in the cw-train repo).
- The decoder is cross-validated with the Go decoder in gocw and the PyTorch
  reference through the shared `golden.bin` contract.