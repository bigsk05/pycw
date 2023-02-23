import os

from argparse import ArgumentParser

from . import output_wave

def main():
    parser = ArgumentParser(
        description='Generate Morse Code (CW) audio files in Python.'
    )
    parser.add_argument('-i', '--input', default="-",
                help='Input text file (defaults to stdin)')
    parser.add_argument('-t', '--text',
                help='Input text. Overrides --input.')
    parser.add_argument('-s', '--speed', type=int, default=12,
                help='Speed, in words per minute (default: 12)')
    parser.add_argument('-n', '--tone', type=int, default=800,
                help='Tone frequency, in Hz (default: 800)')
    parser.add_argument('-v', '--volume', type=float, default=1.0,
                help='Volume (default: 1.0)')
    parser.add_argument('-r', '--sample_rate', type=int, default=44100,
                help='Sample rate (default: 44100)')
    parser.add_argument('-o', '--output',
                help='Name of the output file')
    args = parser.parse_args()

    if not args.input:
        with open(args.input, "r") as fb:
            input_text = fb.read()
    else:
        input_text = args.text

    if input_text == "-" or input_text is None:
        os.system("pycw -h")
    else:
        output_wave(
            args.output, input_text, 
            args.speed, args.tone, args.volume,
            args.sample_rate
        )

if __name__ == '__main__':
    main()
