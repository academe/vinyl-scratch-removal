#!/usr/bin/env python3
"""
Example: Periodicity-based click detection for vinyl scratch removal.

This demonstrates how to use the rotation period of a vinyl record
to distinguish real scratches (periodic) from random noise (dust, static).
"""

import numpy as np
import matplotlib.pyplot as plt


class PeriodicityDetector:
    """Detect periodic clicks from vinyl scratches."""

    # Standard RPM values and their periods
    RPM_PERIODS = {
        33.333: 60.0 / 33.333,  # 1.800 seconds
        45.0: 60.0 / 45.0,       # 1.333 seconds
        78.0: 60.0 / 78.0        # 0.769 seconds
    }

    def __init__(self, sample_rate=44100):
        """
        Initialize detector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def detect_rpm(self, click_positions):
        """
        Automatically detect record RPM from click periodicity.

        Args:
            click_positions: Array of click positions in samples

        Returns:
            rpm: Detected RPM (33.333, 45.0, 78.0, or None)
            confidence: Detection confidence (0-1)
        """
        if len(click_positions) < 5:
            return None, 0.0

        best_rpm = None
        best_confidence = 0.0

        for rpm, period_sec in self.RPM_PERIODS.items():
            expected_period = int(period_sec * self.sample_rate)
            confidence = self._check_period(click_positions, expected_period)

            if confidence > best_confidence:
                best_confidence = confidence
                best_rpm = rpm

        return best_rpm, best_confidence

    def _check_period(self, positions, expected_period):
        """
        Check how well positions match expected period.

        Args:
            positions: Click positions in samples
            expected_period: Expected period in samples

        Returns:
            Fraction of clicks matching period (0-1)
        """
        tolerance = int(0.05 * expected_period)  # ±5%

        matches = 0
        for pos in positions:
            # Find nearest multiple of period
            n = round(pos / expected_period)
            expected_pos = n * expected_period

            if abs(pos - expected_pos) < tolerance:
                matches += 1

        return matches / len(positions)

    def classify_clicks(self, click_positions, rpm=None):
        """
        Classify clicks as periodic or random.

        Args:
            click_positions: Array of click positions in samples
            rpm: Known RPM, or None to auto-detect

        Returns:
            classifications: List of 'periodic' or 'random'
            detected_rpm: Detected RPM
            confidence: Detection confidence
        """
        # Auto-detect RPM if not provided
        if rpm is None:
            rpm, confidence = self.detect_rpm(click_positions)
            if rpm is None:
                return ['random'] * len(click_positions), None, 0.0
        else:
            confidence = 1.0

        period_sec = self.RPM_PERIODS[rpm]
        expected_period = int(period_sec * self.sample_rate)
        tolerance = int(0.05 * expected_period)

        classifications = []
        for pos in click_positions:
            n = round(pos / expected_period)
            expected_pos = n * expected_period

            if abs(pos - expected_pos) < tolerance:
                classifications.append('periodic')
            else:
                classifications.append('random')

        return classifications, rpm, confidence


def create_synthetic_vinyl(duration=10.0, sample_rate=44100, rpm=33.333):
    """
    Create synthetic vinyl recording with periodic scratches.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        rpm: Record speed (33.333, 45, or 78)

    Returns:
        audio: Audio array with scratches and noise
        true_scratch_positions: Actual scratch positions
    """
    n_samples = int(duration * sample_rate)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.05

    # Add periodic scratches
    period_sec = 60.0 / rpm
    period_samples = int(period_sec * sample_rate)

    scratch_positions = []
    for i in range(int(duration / period_sec)):
        pos = int(i * period_samples + np.random.randn() * 100)  # Small jitter
        if pos < n_samples - 10:
            audio[pos:pos + 5] += np.random.randn(5) * 2.0  # Large amplitude
            scratch_positions.append(pos)

    # Add random dust clicks
    n_dust = 20
    for _ in range(n_dust):
        pos = np.random.randint(0, n_samples - 10)
        audio[pos:pos + 3] += np.random.randn(3) * 1.0

    return audio, scratch_positions


def demo_periodicity_detection():
    """Demonstrate periodicity detection."""
    print("=" * 60)
    print("Vinyl Scratch Periodicity Detection Demo")
    print("=" * 60)
    print()

    # Create synthetic vinyl recording
    print("Creating synthetic vinyl recording...")
    rpm = 33.333
    duration = 10.0
    sample_rate = 44100

    audio, true_scratches = create_synthetic_vinyl(duration, sample_rate, rpm)
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  RPM: {rpm}")
    print(f"  True scratches: {len(true_scratches)}")
    print()

    # Detect all clicks (simple threshold detection)
    print("Detecting all clicks...")
    threshold = 0.5
    detected_clicks = []

    for i in range(2, len(audio)):
        diff2 = abs(audio[i] - 2*audio[i-1] + audio[i-2])
        if diff2 > threshold:
            # Avoid duplicates
            if not detected_clicks or i - detected_clicks[-1] > 10:
                detected_clicks.append(i)

    print(f"  Detected {len(detected_clicks)} clicks (periodic + random)")
    print()

    # Analyze periodicity
    print("Analyzing periodicity...")
    detector = PeriodicityDetector(sample_rate)

    classifications, detected_rpm, confidence = detector.classify_clicks(
        np.array(detected_clicks)
    )

    periodic_count = classifications.count('periodic')
    random_count = classifications.count('random')

    print(f"  Detected RPM: {detected_rpm} (confidence: {confidence:.1%})")
    print(f"  Periodic clicks: {periodic_count}")
    print(f"  Random clicks: {random_count}")
    print()

    # Show results
    print("Classification results:")
    print("  " + "-" * 50)
    print(f"  {'Position':>10} {'Classification':>15} {'Expected?':>12}")
    print("  " + "-" * 50)

    for i, (pos, classification) in enumerate(zip(detected_clicks[:20], classifications[:20])):
        # Check if this was a true scratch
        is_true = any(abs(pos - true_pos) < 200 for true_pos in true_scratches)
        correct = "✓" if (classification == 'periodic') == is_true else "✗"
        print(f"  {pos:10d} {classification:>15} {correct:>12}")

    if len(detected_clicks) > 20:
        print(f"  ... ({len(detected_clicks) - 20} more)")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"True scratches added: {len(true_scratches)}")
    print(f"Periodic clicks detected: {periodic_count}")
    print(f"Random clicks detected: {random_count}")
    print()

    accuracy = periodic_count / len(true_scratches) if true_scratches else 0
    print(f"Detection accuracy: {accuracy:.1%}")
    print()

    print("Key insight:")
    print("  Periodic clicks → Real scratches (high confidence)")
    print("  Random clicks → Dust/noise (lower confidence)")
    print()


def demo_visualization():
    """Create visualization of periodicity."""
    print("Creating visualization...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Matplotlib not available, skipping visualization")
        return

    # Create test data
    rpm = 45.0
    duration = 10.0
    sample_rate = 44100

    audio, _ = create_synthetic_vinyl(duration, sample_rate, rpm)

    # Detect clicks
    threshold = 0.5
    detected_clicks = []
    for i in range(2, len(audio)):
        diff2 = abs(audio[i] - 2*audio[i-1] + audio[i-2])
        if diff2 > threshold:
            if not detected_clicks or i - detected_clicks[-1] > 10:
                detected_clicks.append(i)

    # Classify
    detector = PeriodicityDetector(sample_rate)
    classifications, detected_rpm, confidence = detector.classify_clicks(
        np.array(detected_clicks)
    )

    # Calculate phases
    period = (60.0 / rpm) * sample_rate
    phases = [(pos % period) / period for pos in detected_clicks]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Click positions over time
    ax = axes[0]
    times = [pos / sample_rate for pos in detected_clicks]
    colors = ['red' if c == 'periodic' else 'blue' for c in classifications]

    ax.scatter(times, [1]*len(times), c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title(f'Click Positions (Red=Periodic, Blue=Random)\nDetected RPM: {detected_rpm} (confidence: {confidence:.1%})')
    ax.grid(True, alpha=0.3)

    # Add period markers
    for i in range(int(duration / (60.0 / rpm))):
        ax.axvline(i * 60.0 / rpm, color='green', alpha=0.3, linestyle='--')

    # Plot 2: Phase histogram
    ax = axes[1]
    periodic_phases = [p for p, c in zip(phases, classifications) if c == 'periodic']
    random_phases = [p for p, c in zip(phases, classifications) if c == 'random']

    ax.hist(periodic_phases, bins=50, range=(0, 1), alpha=0.7, color='red', label='Periodic')
    ax.hist(random_phases, bins=50, range=(0, 1), alpha=0.7, color='blue', label='Random')
    ax.set_xlabel('Phase (0 = start of rotation, 1 = end of rotation)')
    ax.set_ylabel('Count')
    ax.set_title('Click Phase Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('periodicity_demo.png', dpi=150)
    print("  Saved visualization to: periodicity_demo.png")
    print()


if __name__ == '__main__':
    # Run demo
    demo_periodicity_detection()

    # Create visualization
    demo_visualization()

    print("=" * 60)
    print("Demo complete!")
    print()
    print("See docs/PERIODICITY_DETECTION.md for full documentation")
    print("=" * 60)
