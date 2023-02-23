import numpy

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_VOLUME = 1.0

def generate_sin_wave(
        frequency: int, 
        duration: float,
        volume: float = DEFAULT_VOLUME,
        attack: int = 0,
        release: int = 0,
        sample_rate: int = int(DEFAULT_SAMPLE_RATE)
    ):
    samples = volume * numpy.sin(
        2 * numpy.pi *
        numpy.arange(sample_rate * duration) *
        frequency / sample_rate,
    ).astype(numpy.float32)

    if (attack + release) > duration:
        raise ValueError('Attack + release times cannot be > total time')

    if attack > 0:
        attack_samples_num = int(attack * sample_rate)
        attack_samples = samples[:attack_samples_num]
        other_samples = samples[attack_samples_num:]
        attack_envelope = (
            numpy.arange(
                attack_samples_num
            ) / attack_samples_num
        )
        samples = numpy.concatenate((
            (attack_samples * attack_envelope), other_samples
        ))

    if release > 0:
        release_samples_num = int(release * sample_rate)
        release_samples = samples[-release_samples_num:]
        other_samples = samples[:-release_samples_num]
        release_envelope = (
            numpy.arange(
                release_samples_num - 1, -1, -1
            ) / release_samples_num
        )
        samples = numpy.concatenate((
            other_samples, (release_samples * release_envelope)
        ))

    return (samples * (2**15 - 1) / numpy.max(numpy.abs(samples))).astype(numpy.int16)


def generate_silence(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE):
    return numpy.zeros(int(duration * sample_rate)).astype(numpy.int16)