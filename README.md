# pycw

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

pycw.output_wave("Intro.wav", "CQ CQ CQ DE BG5AWO BG5AWO BG5AWO PSE K", 20)
```

Then you can get a output file called `Intro.wav` in your working directory.
