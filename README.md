# pycw

![Project Version](https://img.shields.io/pypi/v/pycw) ![Python Version](https://img.shields.io/pypi/pyversions/pycw)

Python Morse Code Generator

Generate Morse Code (CW) audio files in Python.

## Usage

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
```

Or `import pycw` and then use functions in your code, for example:

```
import pycw

pycw.output_wave("Intro.wav", "CQ CQ CQ DE BD8CMN BD8CMN BD8CMN PSE K", 20)
```

Then you can get a output file called `Intro.wav` in your working directory.

## Decoding CW audio (neural decoder)

`pycw` now also decodes Morse audio back to text. The decoder is **numpy-only**
(no PyTorch at runtime); the neural weights (`decoder/model.bin`, ~374KB) ship
inside the package and come from the [cw-train](https://git-direct.ghink.net/bigsk/cw-train)
repository (model.bin is the cross-runtime contract artifact; pycw, gocw and
torch all reproduce the same golden vectors bit-for-bit).

```bash
pycw -d recording.wav            # auto-detects tone, any sample rate
pycw -d recording.wav --decoder-tone 700
```

```python
import pycw
pycw.decode_wav("recording.wav")            # file -> text
pycw.decode_samples(samples, 16000)         # float32 mono in [-1,1] -> text
pycw.decode_bytes(pcm_bytes, 16000)         # raw int16 PCM -> text
```

Notes:

- Robust to background noise, fading, hand-key jitter, QRM: CER ~0.02–0.05 at
  -8…+12 dB SNR (see cw-train README for the full measurement table).
- `pip install pycw[train]` adds torch **only** if you want to retrain models
  (training itself lives in the cw-train repo).
- Golden parity tests: `python tests/test_decoder.py`.
