# cython: language_level=3
# vinyl_core.pyx - Main vinyl scratch removal processor

import numpy as np
cimport numpy as np
from detection cimport detect_clicks_internal
from interpolation cimport interpolate_cubic_internal
from detection import detect_clicks_python, calculate_rms
from interpolation import interpolate_cubic_python, interpolate_ar_python, blend_interpolation
from libc.stdlib cimport malloc, free

ctypedef np.float32_t DTYPE_t

# Detection modes
DEF MODE_CONSERVATIVE = 0
DEF MODE_STANDARD = 1
DEF MODE_AGGRESSIVE = 2

cdef struct VinylProcessorState:
    float sample_rate
    float threshold
    int ar_order
    int mode
    int max_width_samples


cdef class VinylProcessor:
    """
    Main processor class for vinyl scratch removal.
    """
    cdef VinylProcessorState state

    def __init__(self, float sample_rate=44100.0,
                 float threshold=3.0,
                 int ar_order=20,
                 int mode=MODE_STANDARD):
        """
        Initialize processor.

        Args:
            sample_rate: Audio sample rate in Hz
            threshold: Detection threshold (std deviations)
            ar_order: Order of AR model for interpolation
            mode: Detection mode (0=conservative, 1=standard, 2=aggressive)
        """
        self.state.sample_rate = sample_rate
        self.state.threshold = threshold
        self.state.ar_order = ar_order
        self.state.mode = mode
        self.state.max_width_samples = int(0.002 * sample_rate)  # 2ms default

    def set_threshold(self, float threshold):
        """Set detection threshold."""
        self.state.threshold = threshold

    def set_mode(self, int mode):
        """Set detection mode (0-2)."""
        self.state.mode = mode

    def set_max_width_ms(self, float width_ms):
        """Set maximum click width in milliseconds."""
        self.state.max_width_samples = int(width_ms * 0.001 * self.state.sample_rate)

    def detect(self, np.ndarray[DTYPE_t, ndim=1] audio):
        """
        Detect clicks in audio.

        Returns:
            List of (start, end) tuples for detected clicks
        """
        cdef int window_size = int(0.01 * self.state.sample_rate)  # 10ms
        cdef float threshold = self.state.threshold

        # Adjust threshold based on mode
        if self.state.mode == MODE_CONSERVATIVE:
            threshold *= 1.5
        elif self.state.mode == MODE_AGGRESSIVE:
            threshold *= 0.7

        return detect_clicks_python(
            audio,
            threshold,
            self.state.max_width_samples,
            window_size
        )

    def process(self, np.ndarray[DTYPE_t, ndim=1] audio):
        """
        Process audio to remove clicks.

        Args:
            audio: Input audio (will be modified in-place)

        Returns:
            Processed audio
        """
        # Detect clicks
        clicks = self.detect(audio)

        print(f"Detected {len(clicks)} clicks")

        # Interpolate each click
        cdef int i
        cdef int start, end
        cdef np.ndarray[DTYPE_t, ndim=1] interpolated
        cdef np.ndarray[DTYPE_t, ndim=1] window

        for i, (start, end) in enumerate(clicks):
            if i % 100 == 0 and i > 0:
                print(f"Processing click {i}/{len(clicks)}")

            # Use AR interpolation for better quality
            try:
                interpolated = interpolate_ar_python(
                    audio, start, end, self.state.ar_order
                )
            except:
                # Fall back to cubic on error
                interpolated = interpolate_cubic_python(audio, start, end)

            # Create blend window
            window = blend_interpolation(interpolated)

            # Apply with blending
            if len(interpolated) == end - start:
                audio[start:end] = (
                    interpolated * window +
                    audio[start:end] * (1 - window)
                )

        return audio

    def process_file(self, str input_path, str output_path):
        """
        Process audio file.

        Args:
            input_path: Path to input WAV file
            output_path: Path to output WAV file
        """
        import soundfile as sf

        # Load audio
        audio, sample_rate = sf.read(input_path, dtype='float32')

        # Update sample rate
        self.state.sample_rate = sample_rate

        # Process
        if audio.ndim == 1:
            # Mono
            processed = self.process(audio)
        else:
            # Stereo - process each channel
            processed = np.zeros_like(audio)
            print("Processing left channel...")
            processed[:, 0] = self.process(audio[:, 0].copy())
            print("Processing right channel...")
            processed[:, 1] = self.process(audio[:, 1].copy())

        # Save
        sf.write(output_path, processed, int(sample_rate))


def process_audio_simple(np.ndarray[DTYPE_t, ndim=1] audio,
                         float sample_rate=44100.0,
                         float threshold=3.0):
    """
    Simple convenience function for processing audio.

    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        threshold: Detection threshold

    Returns:
        Processed audio
    """
    processor = VinylProcessor(sample_rate, threshold)
    return processor.process(audio.copy())
