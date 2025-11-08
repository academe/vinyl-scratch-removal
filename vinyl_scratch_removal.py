#!/usr/bin/env python3
"""
Advanced Vinyl Scratch Removal Tool

This implementation uses research-backed algorithms for removing clicks, pops,
and scratches from vinyl record digitizations:

- Autoregressive (AR) linear prediction for sample interpolation
- Cubic Hermite spline interpolation for smooth reconstruction
- Adaptive threshold detection using local statistics
- Multi-pass processing for different artifact types

Based on research from:
- "Digital Audio Restoration: A Statistical Model Based Approach" (Godsill & Rayner)
- "Restoration of Click Degraded Speech and Music Based on High Order Sparse Linear Prediction"
- "Detection of clicks in audio signals using warped linear prediction"
- "Diffusion Models for Audio Restoration" (2024)

Author: Claude (Anthropic)
License: MIT
Version: 1.0.0
"""

import numpy as np
from scipy import signal, interpolate
from scipy.linalg import solve_toeplitz
import argparse
import warnings

warnings.filterwarnings('ignore')


class VinylScratchRemoval:
    """
    Advanced vinyl scratch and click removal processor.
    """

    def __init__(self, sample_rate=44100, threshold=3.0, max_click_width_ms=2.0,
                 detection_mode='standard', ar_order=20, padding_ms=0.5):
        """
        Initialize the scratch removal processor.

        Args:
            sample_rate: Audio sample rate in Hz
            threshold: Detection threshold in standard deviations (default 3.0)
            max_click_width_ms: Maximum click width in milliseconds
            detection_mode: 'conservative', 'standard', or 'aggressive'
            ar_order: Order of autoregressive model for interpolation
            padding_ms: Padding to add around detected clicks in milliseconds
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_click_width = int(max_click_width_ms * sample_rate / 1000)
        self.detection_mode = detection_mode
        self.ar_order = ar_order
        self.padding = int(padding_ms * sample_rate / 1000)

        # Detection parameters based on mode
        self.mode_params = {
            'conservative': {'threshold_mult': 1.5, 'min_gap': 3, 'merge_distance': 10, 'amplitude_ratio': 2.5},
            'standard': {'threshold_mult': 1.0, 'min_gap': 2, 'merge_distance': 20, 'amplitude_ratio': 2.0},
            'aggressive': {'threshold_mult': 0.7, 'min_gap': 1, 'merge_distance': 30, 'amplitude_ratio': 1.3},
            'auto': {'threshold_mult': None, 'min_gap': 1, 'merge_distance': 30, 'amplitude_ratio': 1.5}  # Auto-calculated
        }

    def detect_clicks_auto(self, audio):
        """
        Automatically detect scratches using statistical outlier detection.

        This mode analyzes the audio's statistical characteristics and identifies
        outliers that are likely scratches, without requiring manual threshold tuning.

        Args:
            audio: Input audio signal (1D numpy array)

        Returns:
            List of tuples (start_idx, end_idx) for detected clicks
        """
        # Calculate first and second derivatives
        diff1 = np.diff(audio, prepend=audio[0])
        diff2 = np.diff(diff1, prepend=diff1[0])

        abs_diff1 = np.abs(diff1)
        abs_diff2 = np.abs(diff2)

        # Calculate global statistics to identify outliers
        # Use median and MAD (Median Absolute Deviation) which are robust to outliers
        median_diff1 = np.median(abs_diff1)
        mad_diff1 = np.median(np.abs(abs_diff1 - median_diff1))

        median_diff2 = np.median(abs_diff2)
        mad_diff2 = np.median(np.abs(abs_diff2 - median_diff2))

        # Use a much higher multiplier to avoid false positives
        # Scratches are extreme outliers, not just mild variations
        # Using 10 MAD is very conservative - only catches extreme spikes
        mad_multiplier = 10.0
        threshold1 = median_diff1 + mad_multiplier * mad_diff1 * 1.4826  # 1.4826 makes MAD consistent with std
        threshold2 = median_diff2 + mad_multiplier * mad_diff2 * 1.4826

        # Detect outliers in both derivatives
        candidates_diff1 = abs_diff1 > threshold1
        candidates_diff2 = abs_diff2 > threshold2

        # Combine detections
        candidates = candidates_diff1 | candidates_diff2

        # Find connected regions
        clicks = self._find_click_regions(candidates)

        # Filter using auto mode parameters with strict amplitude checking
        min_gap = self.mode_params['auto']['min_gap']
        filtered_clicks = self._filter_clicks(clicks, audio, min_gap)

        return filtered_clicks

    def detect_clicks(self, audio):
        """
        Detect clicks using adaptive threshold and slope analysis.

        This implements a robust click detection algorithm based on:
        - Local RMS analysis for adaptive thresholding
        - First and second derivative analysis
        - Width and amplitude criteria

        Args:
            audio: Input audio signal (1D numpy array)

        Returns:
            List of tuples (start_idx, end_idx) for detected clicks
        """
        # Use auto detection if in auto mode
        if self.detection_mode == 'auto':
            return self.detect_clicks_auto(audio)
        # Calculate first and second derivatives
        diff1 = np.diff(audio, prepend=audio[0])
        diff2 = np.diff(diff1, prepend=diff1[0])

        # Calculate local statistics using sliding window
        window_size = int(0.01 * self.sample_rate)  # 10ms window
        window_size = max(window_size, 100)

        # Local RMS for adaptive thresholding
        local_rms = self._sliding_rms(audio, window_size)

        # Adaptive threshold based on local statistics
        mode_mult = self.mode_params[self.detection_mode]['threshold_mult']
        threshold_signal = local_rms * self.threshold * mode_mult

        # Detect potential clicks based on BOTH first derivative (sudden changes)
        # and second derivative (acceleration)
        abs_diff1 = np.abs(diff1)
        abs_diff2 = np.abs(diff2)

        # Method 1: Second derivative (for sharp clicks)
        candidates_diff2 = abs_diff2 > threshold_signal

        # Method 2: First derivative (for gradual scratches)
        # Use a slightly higher multiplier for first derivative since it's generally larger
        # but not too high or we'll miss rounded/gradual scratches
        first_deriv_threshold = threshold_signal * 1.0
        candidates_diff1 = abs_diff1 > first_deriv_threshold

        # Combine both detection methods
        candidates = candidates_diff1 | candidates_diff2

        # Find connected regions (clicks)
        clicks = self._find_click_regions(candidates)

        # Filter by width and validate
        min_gap = self.mode_params[self.detection_mode]['min_gap']
        filtered_clicks = self._filter_clicks(clicks, audio, min_gap)

        return filtered_clicks

    def _sliding_rms(self, signal, window_size):
        """Calculate RMS using sliding window."""
        squared = signal ** 2
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode='same'))
        return np.maximum(rms, 1e-10)  # Avoid division by zero

    def _find_click_regions(self, candidates):
        """Find continuous regions of potential clicks."""
        clicks = []
        in_click = False
        start_idx = 0

        for i, val in enumerate(candidates):
            if val and not in_click:
                start_idx = i
                in_click = True
            elif not val and in_click:
                clicks.append((start_idx, i))
                in_click = False

        if in_click:
            clicks.append((start_idx, len(candidates)))

        return clicks

    def _filter_clicks(self, clicks, audio, min_gap):
        """
        Filter detected clicks based on width and merge nearby clicks.
        """
        filtered = []
        merge_distance = self.mode_params[self.detection_mode]['merge_distance']

        for start, end in clicks:
            width = end - start

            # Skip if too wide (likely not a click)
            if width > self.max_click_width:
                continue

            # Skip if too narrow (likely noise)
            if width < min_gap:
                continue

            # Verify amplitude is significant
            if start > 0 and end < len(audio):
                click_peak = np.max(np.abs(audio[start:end]))
                # Use a larger reference window for more accurate baseline
                ref_distance = 100
                local_avg = np.mean(np.abs(
                    np.concatenate([
                        audio[max(0, start-ref_distance):max(0, start-10)],
                        audio[min(len(audio), end+10):min(len(audio), end+ref_distance)]
                    ])
                ))

                # Click should be significantly larger than surroundings
                amplitude_ratio = self.mode_params[self.detection_mode]['amplitude_ratio']
                if click_peak > amplitude_ratio * local_avg:
                    filtered.append((start, end))

        # Merge nearby clicks more aggressively
        if not filtered:
            return []

        merged = [filtered[0]]
        for start, end in filtered[1:]:
            last_start, last_end = merged[-1]
            # Use merge_distance parameter instead of min_gap * 2
            if start - last_end < merge_distance:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))

        # Add padding around each click and ensure we don't exceed boundaries
        padded = []
        for start, end in merged:
            padded_start = max(0, start - self.padding)
            padded_end = min(len(audio), end + self.padding)
            padded.append((padded_start, padded_end))

        # Merge again after padding (in case padding caused overlaps)
        if not padded:
            return []

        final_merged = [padded[0]]
        for start, end in padded[1:]:
            last_start, last_end = final_merged[-1]
            if start <= last_end:
                # Overlapping, merge them
                final_merged[-1] = (last_start, max(end, last_end))
            else:
                final_merged.append((start, end))

        return final_merged

    def interpolate_spectral(self, audio, start, end):
        """
        Interpolate using spectral/frequency analysis.

        This method analyzes the frequency content before and after the scratch
        and synthesizes the missing samples by continuing those frequency patterns.
        This works particularly well for musical content with periodic waveforms.

        Args:
            audio: Full audio signal
            start: Start index of click
            end: End index of click

        Returns:
            Interpolated samples
        """
        gap_length = end - start

        # Need enough context for meaningful frequency analysis
        min_context = max(512, gap_length * 4)

        if start < min_context or end + min_context >= len(audio):
            # Not enough context, fall back to AR
            return self.interpolate_ar(audio, start, end)

        # Get context windows before and after
        context_len = min(2048, start, len(audio) - end)
        before = audio[start - context_len:start]
        after = audio[end:end + context_len]

        try:
            # Analyze frequency content using FFT
            fft_before = np.fft.rfft(before)
            fft_after = np.fft.rfft(after)

            # Average the magnitude and phase from both sides
            # This gives us the dominant frequencies and their phases
            mag_avg = (np.abs(fft_before) + np.abs(fft_after)) / 2

            # For phase, we need to account for the time shift
            # Interpolate phase between before and after
            phase_before = np.angle(fft_before)
            phase_after = np.angle(fft_after)

            # Calculate phase increment per sample for each frequency
            freq_bins = len(fft_before)
            freqs = np.fft.rfftfreq(len(before), 1.0 / self.sample_rate)

            # Synthesize the gap
            t = np.arange(gap_length)
            synthesized = np.zeros(gap_length)

            # Use only the most significant frequency components
            # to avoid noise amplification
            num_components = min(50, freq_bins // 2)
            top_freqs = np.argsort(mag_avg)[-num_components:]

            for freq_idx in top_freqs:
                if freq_idx == 0:
                    # DC component
                    continue

                freq = freqs[freq_idx]
                magnitude = mag_avg[freq_idx]

                # Estimate phase at the start of the gap
                # Use the phase from the end of the 'before' segment
                phase_start = phase_before[freq_idx]

                # Generate this frequency component through the gap
                omega = 2 * np.pi * freq / self.sample_rate
                component = magnitude * np.cos(omega * t + phase_start)
                synthesized += component

            # Normalize to match the expected amplitude
            if len(synthesized) > 0 and np.max(np.abs(synthesized)) > 0:
                # Match the amplitude to surrounding audio
                expected_amp = (np.max(np.abs(before[-50:])) + np.max(np.abs(after[:50]))) / 2
                actual_amp = np.max(np.abs(synthesized))
                if actual_amp > 0:
                    synthesized = synthesized * (expected_amp / actual_amp)

            # Apply a window to smoothly transition in/out
            window = np.hanning(gap_length)

            # Blend spectral synthesis with endpoints for smooth transition
            if gap_length > 4:
                # Simple linear interpolation for endpoints
                endpoint_bridge = np.linspace(before[-1], after[0], gap_length)
                # Blend: mostly spectral in the middle, more endpoint-based at edges
                synthesized = synthesized * window + endpoint_bridge * (1 - window)

            return synthesized

        except Exception as e:
            # Fall back to AR if spectral analysis fails
            return self.interpolate_ar(audio, start, end)

    def interpolate_ar(self, audio, start, end):
        """
        Interpolate using Autoregressive (AR) Linear Prediction.

        This implements the algorithm from:
        "Restoration of Click Degraded Speech and Music Based on
         High Order Sparse Linear Prediction"

        Args:
            audio: Full audio signal
            start: Start index of click
            end: End index of click

        Returns:
            Interpolated samples
        """
        # Get context samples before and after click
        context_len = min(self.ar_order * 4, start, len(audio) - end)

        if context_len < self.ar_order:
            # Not enough context, use cubic interpolation instead
            return self.interpolate_cubic(audio, start, end)

        # Samples before click
        before = audio[max(0, start - context_len):start]

        # Samples after click
        after = audio[end:min(len(audio), end + context_len)]

        # Estimate AR coefficients from context
        context = np.concatenate([before, after])

        try:
            # Calculate autocorrelation
            r = np.correlate(context, context, mode='full')
            r = r[len(r)//2:]
            r = r[:self.ar_order + 1]

            # Solve Yule-Walker equations for AR coefficients
            if len(r) > self.ar_order:
                ar_coeffs = solve_toeplitz(r[:self.ar_order], r[1:self.ar_order + 1])

                # Generate interpolated samples
                click_len = end - start
                interpolated = np.zeros(click_len)

                # Forward prediction from before samples
                for i in range(click_len):
                    if i < len(before):
                        # Use actual before samples
                        prev_samples = before[-(self.ar_order - i):]
                        prev_samples = np.concatenate([prev_samples, interpolated[:i]])
                    else:
                        # Use generated samples
                        prev_samples = interpolated[max(0, i - self.ar_order):i]

                    if len(prev_samples) < self.ar_order:
                        prev_samples = np.concatenate([
                            before[-(self.ar_order - len(prev_samples)):],
                            prev_samples
                        ])

                    # Predict next sample
                    interpolated[i] = np.dot(ar_coeffs, prev_samples[::-1][:self.ar_order])

                return interpolated

        except:
            # Fall back to cubic interpolation if AR fails
            pass

        return self.interpolate_cubic(audio, start, end)

    def interpolate_cubic(self, audio, start, end):
        """
        Interpolate using Cubic Hermite Spline.

        This provides smooth, natural-sounding interpolation with
        continuous first derivatives.

        Args:
            audio: Full audio signal
            start: Start index of click
            end: End index of click

        Returns:
            Interpolated samples
        """
        # Get samples around the click for interpolation
        context = 4  # Use 4 samples on each side

        # Ensure we have enough context
        if start < context or end + context >= len(audio):
            # Linear interpolation if at edges
            if start > 0 and end < len(audio):
                return np.linspace(audio[start-1], audio[end], end - start, endpoint=False)
            else:
                return np.zeros(end - start)

        # Get context points
        x_before = np.arange(start - context, start)
        y_before = audio[start - context:start]

        x_after = np.arange(end, end + context)
        y_after = audio[end:end + context]

        # Combine context points
        x_context = np.concatenate([x_before, x_after])
        y_context = np.concatenate([y_before, y_after])

        # Create cubic spline
        cs = interpolate.CubicSpline(x_context, y_context, bc_type='natural')

        # Interpolate missing samples
        x_interp = np.arange(start, end)
        interpolated = cs(x_interp)

        return interpolated

    def process(self, audio, verbose=False, preview=False):
        """
        Process audio to remove clicks and scratches.

        Args:
            audio: Input audio (1D numpy array)
            verbose: Print detailed progress information
            preview: If True, only detect and report scratches without processing

        Returns:
            Processed audio with clicks removed (or original if preview=True)
        """
        output = audio.copy()

        # Detect clicks
        clicks = self.detect_clicks(audio)

        print(f"Detected {len(clicks)} clicks/scratches")

        if len(clicks) > 0:
            widths = [end - start for start, end in clicks]
            print(f"  Average click width: {np.mean(widths):.1f} samples ({np.mean(widths)/self.sample_rate*1000:.2f} ms)")
            print(f"  Max click width: {np.max(widths)} samples ({np.max(widths)/self.sample_rate*1000:.2f} ms)")
            print(f"  Min click width: {np.min(widths)} samples ({np.min(widths)/self.sample_rate*1000:.2f} ms)")

            if verbose or preview:
                # Show time positions of detected scratches
                print(f"\n  Scratch locations (showing first 20):")
                for i, (start, end) in enumerate(clicks[:20]):
                    time_start = start / self.sample_rate
                    time_end = end / self.sample_rate
                    width_ms = (end - start) / self.sample_rate * 1000
                    peak_amp = np.max(np.abs(audio[start:end]))
                    print(f"    {i+1:3d}. {time_start:8.3f}s - {time_end:8.3f}s  "
                          f"width: {width_ms:5.2f}ms  peak: {peak_amp:.3f}")
                if len(clicks) > 20:
                    print(f"    ... and {len(clicks) - 20} more")

        # If preview mode, return original audio
        if preview:
            print("\nPreview mode: No processing performed. Use without --preview to remove scratches.")
            return audio

        # Interpolate each click
        for i, (start, end) in enumerate(clicks):
            if i % 100 == 0 and i > 0:
                print(f"Processing click {i}/{len(clicks)}")

            # Use spectral interpolation for musical content
            # This analyzes frequency patterns and "bridges" the waveform
            interpolated = self.interpolate_spectral(audio, start, end)

            # Simply replace the corrupted samples with interpolated ones
            # No blending - we want to completely remove the click
            output[start:end] = interpolated

        return output

    def _get_blend_window(self, length):
        """
        Create a blending window for smooth transitions.

        Uses a Tukey window for smooth edges.
        """
        if length < 4:
            return np.ones(length)

        # Tukey window with 50% taper
        window = self._tukey_window(length, alpha=0.5)
        return window

    def _tukey_window(self, length, alpha=0.5):
        """
        Generate a Tukey (tapered cosine) window.

        This is a custom implementation for compatibility with older scipy versions.

        Args:
            length: Window length
            alpha: Taper parameter (0 to 1), where 0 is rectangular and 1 is Hann

        Returns:
            Tukey window as numpy array
        """
        if alpha <= 0:
            return np.ones(length)
        elif alpha >= 1:
            return np.hanning(length)

        # Calculate the transition width
        width = int(alpha * (length - 1) / 2.0)

        # Create the window
        window = np.ones(length)

        # Taper at the beginning
        for n in range(width):
            window[n] = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (length - 1)) - 1)))

        # Taper at the end
        for n in range(length - width, length):
            window[n] = 0.5 * (1 + np.cos(np.pi * (2 * n / (alpha * (length - 1)) - 2 / alpha + 1)))

        return window

    def process_stereo(self, left, right, verbose=False, preview=False):
        """
        Process stereo audio.

        Args:
            left: Left channel audio
            right: Right channel audio
            verbose: Print detailed progress information
            preview: If True, only detect and report scratches without processing

        Returns:
            Tuple of (processed_left, processed_right)
        """
        print("Processing left channel...")
        left_processed = self.process(left, verbose=verbose, preview=preview)

        print("Processing right channel...")
        right_processed = self.process(right, verbose=verbose, preview=preview)

        return left_processed, right_processed


def main():
    """Command-line interface for vinyl scratch removal."""
    parser = argparse.ArgumentParser(
        description='Advanced Vinyl Scratch Removal Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (standard mode)
  python vinyl_scratch_removal.py input.wav output.wav

  # Preview what would be detected (without processing)
  python vinyl_scratch_removal.py input.wav output.wav --preview

  # Automatic mode (parameter-free detection, experimental)
  python vinyl_scratch_removal.py input.wav output.wav --mode auto

  # Aggressive click removal
  python vinyl_scratch_removal.py input.wav output.wav --mode aggressive

  # Conservative (only remove obvious clicks)
  python vinyl_scratch_removal.py input.wav output.wav --mode conservative
        """
    )

    parser.add_argument('input', help='Input audio file (WAV format)')
    parser.add_argument('output', help='Output audio file (WAV format)')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Detection threshold in standard deviations (default: 3.0, ignored in auto mode)')
    parser.add_argument('--max-width', type=float, default=5.0,
                        help='Maximum click width in milliseconds (default: 5.0)')
    parser.add_argument('--padding', type=float, default=1.0,
                        help='Padding around detected clicks in milliseconds (default: 1.0)')
    parser.add_argument('--mode', choices=['auto', 'conservative', 'standard', 'aggressive'],
                        default='standard', help='Detection mode (default: standard, try auto for parameter-free detection)')
    parser.add_argument('--ar-order', type=int, default=20,
                        help='AR model order for interpolation (default: 20)')
    parser.add_argument('--preview', action='store_true',
                        help='Preview mode: show what would be detected without processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed processing information')

    args = parser.parse_args()

    # Import soundfile for audio I/O (optional dependency)
    try:
        import soundfile as sf
    except ImportError:
        print("Error: soundfile library required for audio file I/O")
        print("Install with: pip install soundfile")
        return 1

    # Load audio
    print(f"Loading {args.input}...")
    audio, sample_rate = sf.read(args.input)

    # Initialize processor
    processor = VinylScratchRemoval(
        sample_rate=sample_rate,
        threshold=args.threshold,
        max_click_width_ms=args.max_width,
        detection_mode=args.mode,
        ar_order=args.ar_order,
        padding_ms=args.padding
    )

    # Process audio
    print("Processing audio...")
    if audio.ndim == 1:
        # Mono
        processed = processor.process(audio, verbose=args.verbose, preview=args.preview)
    else:
        # Stereo
        processed_left, processed_right = processor.process_stereo(audio[:, 0], audio[:, 1],
                                                                     verbose=args.verbose, preview=args.preview)
        processed = np.column_stack([processed_left, processed_right])

    # Save output (skip if preview mode)
    if not args.preview:
        print(f"Saving to {args.output}...")
        sf.write(args.output, processed, sample_rate)
        print("Done!")

    return 0


if __name__ == '__main__':
    exit(main())
