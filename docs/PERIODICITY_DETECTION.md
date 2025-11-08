# Periodicity-Based Click Detection

## The Insight: Scratches Repeat

Physical scratches on vinyl records produce clicks at **regular intervals** corresponding to the record's rotation period. This characteristic can dramatically improve detection accuracy.

## Table of Contents

1. [The Physics](#the-physics)
2. [Rotation Periods](#rotation-periods)
3. [Detection Algorithm](#detection-algorithm)
4. [Implementation Strategy](#implementation-strategy)
5. [Advantages and Limitations](#advantages-and-limitations)
6. [Research Background](#research-background)
7. [Example Implementation](#example-implementation)

---

## The Physics

### Why Scratches Repeat

**Physical scratches** in the vinyl groove are stationary defects:
- They occupy a fixed position on the record
- As the record rotates, the stylus passes over the same scratch every rotation
- This produces clicks at perfectly regular intervals

**Contrast with non-periodic noise**:
- **Dust particles**: Random, removed by stylus or fall away
- **Static discharge**: Random timing
- **Electronic noise**: No correlation with rotation
- **Stylus issues**: May have own periodicity (different from record)

### Types of Scratches

**Radial scratches** (across grooves):
- Most common from poor handling
- Create periodic clicks every rotation
- Same position in stereo channels

**Circumferential scratches** (along groove):
- Less common
- Create extended noise, not discrete clicks
- Different detection approach needed

**Deep gouges**:
- Severe damage
- Very loud periodic clicks
- May cause skipping

---

## Rotation Periods

### Standard Speeds

| Speed | Period | Samples @ 44.1kHz | Samples @ 48kHz |
|-------|--------|-------------------|-----------------|
| **33⅓ RPM** | 1.8000 s | 79,380 | 86,400 |
| **45 RPM** | 1.3333 s | 58,800 | 64,000 |
| **78 RPM** | 0.7692 s | 33,923 | 36,923 |

### Calculation

```
Period (seconds) = 60 / RPM

Examples:
  33⅓ RPM: 60 / 33.333 = 1.8 seconds
  45 RPM:  60 / 45      = 1.333 seconds
  78 RPM:  60 / 78      = 0.769 seconds
```

### Actual vs Nominal Speed

**Real-world variation**:
- Belt-driven turntables: ±0.5% typical, up to ±2% if worn
- Direct-drive: ±0.1% typical
- Pitch control: Intentional ±8% range

**Implication**: Search for period in range, not exact value:
- 33⅓: Search 1.75 - 1.85 seconds
- 45: Search 1.30 - 1.37 seconds
- 78: Search 0.75 - 0.79 seconds

---

## Detection Algorithm

### High-Level Approach

```
1. Detect all potential clicks (existing algorithm)
2. Compute click time intervals
3. Analyze for periodicity
4. Classify clicks:
   - Periodic → High confidence (real scratch)
   - Non-periodic → Lower confidence (dust, noise)
5. Apply appropriate processing
```

### Detailed Algorithm

#### Step 1: Initial Click Detection

Use existing detection (second derivative, adaptive threshold):

```python
clicks = detect_clicks(audio, threshold)
# Returns: [(start1, end1), (start2, end2), ...]
```

#### Step 2: Extract Click Positions

```python
click_positions = [start for start, end in clicks]
# In samples: [1000, 79380, 158760, ...]
```

#### Step 3: Compute Inter-Click Intervals (ICI)

```python
intervals = []
for i in range(len(click_positions) - 1):
    interval = click_positions[i+1] - click_positions[i]
    intervals.append(interval)

# Example: [78380, 79380, 79380, 150000, 79380, ...]
```

#### Step 4: Histogram Analysis

```python
# Create histogram of intervals
hist, bins = np.histogram(intervals, bins=100)

# Find peaks
peaks = find_peaks(hist)

# Check if dominant peak aligns with expected period
expected_period_samples = int(1.8 * sample_rate)  # For 33⅓ RPM
tolerance = int(0.05 * expected_period_samples)    # ±5%

for peak in peaks:
    if abs(peak - expected_period_samples) < tolerance:
        period_detected = True
```

#### Step 5: Autocorrelation Method (More Robust)

```python
def detect_periodic_clicks(click_positions, sample_rate, rpm=33.333):
    """
    Detect periodic clicks using autocorrelation.

    Args:
        click_positions: Array of click positions in samples
        sample_rate: Audio sample rate
        rpm: Expected record speed

    Returns:
        periodic_indices: Indices of clicks that are periodic
        period: Detected period in samples
    """
    # Expected period
    expected_period = 60.0 / rpm * sample_rate
    search_min = int(expected_period * 0.95)
    search_max = int(expected_period * 1.05)

    # Create impulse train from click positions
    max_pos = max(click_positions)
    impulse_train = np.zeros(max_pos + 1)
    impulse_train[click_positions] = 1

    # Compute autocorrelation
    autocorr = np.correlate(impulse_train, impulse_train, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only

    # Find peak in expected period range
    search_region = autocorr[search_min:search_max]
    peak_offset = np.argmax(search_region)
    detected_period = search_min + peak_offset
    peak_strength = search_region[peak_offset] / autocorr[0]

    # Threshold for periodicity
    if peak_strength > 0.3:  # At least 30% of clicks are periodic
        # Find which clicks match the period
        periodic_indices = []
        for i, pos in enumerate(click_positions):
            # Check if click is approximately N periods from start
            phase = pos % detected_period
            if phase < 0.1 * detected_period or phase > 0.9 * detected_period:
                periodic_indices.append(i)

        return periodic_indices, detected_period
    else:
        return [], None
```

#### Step 6: Classification

```python
# Classify each click
for i, click in enumerate(clicks):
    if i in periodic_indices:
        click_confidence[i] = 'high'  # Likely real scratch
        click_type[i] = 'periodic'
    else:
        click_confidence[i] = 'low'   # Might be dust/noise
        click_type[i] = 'random'
```

---

## Implementation Strategy

### Strategy 1: Confidence-Based Processing

```python
def process_with_confidence(audio, clicks, confidences):
    for (start, end), confidence in zip(clicks, confidences):
        if confidence == 'high':
            # Use best interpolation (AR prediction)
            interpolated = interpolate_ar(audio, start, end, order=30)
        elif confidence == 'low':
            # Use faster method or skip if threshold not strongly exceeded
            if click_amplitude < 2 * threshold:
                continue  # Don't process marginal detections
            interpolated = interpolate_cubic(audio, start, end)

        audio[start:end] = interpolated
```

### Strategy 2: User Options

Add RPM parameter to plugin:

```
Parameters:
  - Threshold
  - Max width
  - Mode
  - RPM (new): Auto / 33⅓ / 45 / 78
  - Periodic only (new): Yes / No
```

If "Periodic only" = Yes:
- Only process clicks matching rotation period
- Ignore non-periodic clicks
- Reduces false positives

### Strategy 3: Automatic RPM Detection

```python
def detect_rpm(click_positions, sample_rate):
    """Automatically detect record speed from click periodicity."""

    candidates = {
        33.333: 60.0 / 33.333,
        45.0: 60.0 / 45.0,
        78.0: 60.0 / 78.0
    }

    best_rpm = None
    best_score = 0

    for rpm, period_sec in candidates.items():
        period_samples = period_sec * sample_rate

        # Check autocorrelation peak at this period
        score = check_periodicity(click_positions, period_samples)

        if score > best_score:
            best_score = score
            best_rpm = rpm

    if best_score > 0.3:  # Sufficient periodicity detected
        return best_rpm
    else:
        return None  # No clear periodicity
```

---

## Advantages and Limitations

### Advantages

✅ **Reduce false positives**:
- Don't process legitimate musical transients
- Distinguish clicks from drums, plucks, etc.

✅ **Increase confidence**:
- Periodic clicks are definitely scratches
- Can use more aggressive interpolation

✅ **Automatic validation**:
- RPM detection validates that clicks are real
- Helps user confirm correct processing

✅ **Better preservation**:
- Musical transients preserved
- Only process verified scratches

### Limitations

⚠️ **Requires multiple rotations**:
- Need at least 3-4 rotations to detect period
- Short recordings may not have enough data

⚠️ **Speed variation**:
- Wow and flutter affect period
- Belt-driven turntables have more variation
- Need wider tolerance window

⚠️ **Multiple scratches**:
- Different scratches at different positions
- Creates complex periodic pattern
- May need to detect multiple periods

⚠️ **Non-periodic clicks still exist**:
- Dust clicks are random
- Still need to process them (with lower confidence)

⚠️ **Computational cost**:
- Autocorrelation adds processing time
- Need FFT for efficient computation

---

## Research Background

### Published Work

**Godsill & Rayner (1998)** - "Digital Audio Restoration"
- Mentions periodic nature of clicks
- Uses it for validation of detection
- Not primary detection method

**Vaseghi & Rayner (1990)**
- "Detection and suppression of impulsive noise"
- Discusses distinguishing periodic from random noise

**Kauppinen (2002)**
- "Methods for detecting impulsive noise in speech and audio signals"
- Uses time-frequency analysis
- Periodicity as secondary feature

### Commercial Implementations

**iZotope RX "Spectral Repair"**:
- Analyzes spectral content across time
- Can detect periodic patterns
- Uses for validation

**Cedar Audio "Declicker"**:
- Professional restoration hardware
- Likely uses periodicity detection
- Not publicly documented

**ClickRepair (Brian Davies)**:
- Manual mode shows waveform
- Users can visually see periodic clicks
- Semi-automatic verification

---

## Example Implementation

### Complete Periodicity Detector

```python
import numpy as np
from scipy.signal import find_peaks, correlate

class PeriodicityDetector:
    """Detect periodic clicks from vinyl scratches."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

        # Expected periods for standard speeds
        self.rpm_periods = {
            33.333: 60.0 / 33.333,
            45.0: 60.0 / 45.0,
            78.0: 60.0 / 78.0
        }

    def detect_rpm(self, click_positions):
        """
        Automatically detect record RPM from click periodicity.

        Returns:
            rpm: Detected RPM (33.333, 45.0, 78.0, or None)
            confidence: Detection confidence (0-1)
        """
        if len(click_positions) < 5:
            return None, 0.0  # Not enough clicks

        best_rpm = None
        best_confidence = 0.0

        for rpm, period_sec in self.rpm_periods.items():
            expected_period = int(period_sec * self.sample_rate)
            confidence = self._check_period(click_positions, expected_period)

            if confidence > best_confidence:
                best_confidence = confidence
                best_rpm = rpm

        return best_rpm, best_confidence

    def _check_period(self, positions, expected_period):
        """Check how well positions match expected period."""
        # Allow ±5% tolerance
        tolerance = int(0.05 * expected_period)

        # Count clicks near expected multiples of period
        matches = 0
        total = 0

        for i, pos in enumerate(positions):
            # Find nearest multiple of period
            n = round(pos / expected_period)
            expected_pos = n * expected_period

            if abs(pos - expected_pos) < tolerance:
                matches += 1
            total += 1

        return matches / total if total > 0 else 0.0

    def classify_clicks(self, click_positions, rpm=None):
        """
        Classify clicks as periodic or random.

        Args:
            click_positions: Array of click positions in samples
            rpm: Known RPM, or None to auto-detect

        Returns:
            classifications: Array of 'periodic' or 'random'
            detected_rpm: Detected RPM
        """
        if rpm is None:
            rpm, confidence = self.detect_rpm(click_positions)
            if rpm is None:
                # No periodicity detected
                return ['random'] * len(click_positions), None
        else:
            confidence = 1.0

        period_sec = self.rpm_periods[rpm]
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

        return classifications, rpm

    def visualize_periodicity(self, click_positions, rpm=None):
        """
        Create visualization of click periodicity.

        Returns dict with data for plotting.
        """
        import matplotlib.pyplot as plt

        if rpm is None:
            rpm, conf = self.detect_rpm(click_positions)

        if rpm:
            period = self.rpm_periods[rpm] * self.sample_rate

            # Calculate phase of each click
            phases = [(pos % period) / period for pos in click_positions]

            # Histogram of phases
            hist, bins = np.histogram(phases, bins=50, range=(0, 1))

            return {
                'rpm': rpm,
                'phases': phases,
                'hist': hist,
                'bins': bins
            }
        else:
            return None


# Usage example
def process_with_periodicity(audio, sample_rate, threshold):
    """Process audio using periodicity detection."""

    # Detect all clicks
    from detection import detect_clicks_python
    clicks = detect_clicks_python(audio, threshold, max_width=100, window_size=441)

    # Extract positions
    positions = np.array([start for start, end in clicks])

    # Detect periodicity
    detector = PeriodicityDetector(sample_rate)
    classifications, rpm = detector.classify_clicks(positions)

    print(f"Detected RPM: {rpm}")
    print(f"Periodic clicks: {classifications.count('periodic')}")
    print(f"Random clicks: {classifications.count('random')}")

    # Process with different strategies
    from interpolation import interpolate_ar_python, interpolate_cubic_python

    for (start, end), classification in zip(clicks, classifications):
        if classification == 'periodic':
            # High-quality interpolation for verified scratches
            interpolated = interpolate_ar_python(audio, start, end, ar_order=30)
            audio[start:end] = interpolated
        elif classification == 'random':
            # Quick interpolation or skip if marginal
            if np.max(np.abs(audio[start:end])) > threshold * 2:
                interpolated = interpolate_cubic_python(audio, start, end)
                audio[start:end] = interpolated

    return audio, rpm
```

### Integration with Existing Code

```python
# In vinyl_core.pyx, add:

from periodicity import PeriodicityDetector

class VinylProcessor:
    def __init__(self, ..., rpm=None, use_periodicity=True):
        # ... existing init ...
        self.rpm = rpm
        self.use_periodicity = use_periodicity
        self.periodicity_detector = PeriodicityDetector(sample_rate)

    def process(self, audio):
        # Detect clicks
        clicks = self.detect(audio)

        if self.use_periodicity and len(clicks) > 5:
            # Classify by periodicity
            positions = np.array([start for start, end in clicks])
            classifications, detected_rpm = \
                self.periodicity_detector.classify_clicks(positions, self.rpm)

            print(f"Detected RPM: {detected_rpm}")

            # Process with classification
            for (start, end), classification in zip(clicks, classifications):
                if classification == 'periodic':
                    # Verified scratch - use best method
                    interpolated = interpolate_ar_python(
                        audio, start, end, self.ar_order
                    )
                else:
                    # Possible noise - use faster method
                    interpolated = interpolate_cubic_python(audio, start, end)

                audio[start:end] = interpolated
        else:
            # Standard processing without periodicity check
            # ... existing code ...

        return audio
```

---

## Performance Impact

### Computational Cost

**Autocorrelation**:
- Naive: O(n²) where n = audio length
- FFT-based: O(n log n)
- Practical: Negligible for click positions only

**For typical recording**:
- 1000 clicks in 10-minute album
- Autocorrelation on 1000 points: < 1ms
- Total overhead: < 0.1% of processing time

**Conclusion**: Periodicity detection adds minimal overhead

---

## Recommended Implementation Priority

### Phase 1: Detection Only (Easiest)
Add RPM auto-detection and display to user:
```
Detected 127 clicks
Estimated RPM: 33⅓ (confidence: 87%)
Periodic clicks: 89 (70%)
Random clicks: 38 (30%)
```

### Phase 2: Confidence-Based Processing
- Process periodic clicks with AR interpolation (high quality)
- Process random clicks with cubic spline (fast)
- Skip marginal detections if random

### Phase 3: User Control
Add parameters:
- RPM: Auto / 33⅓ / 45 / 78
- Process periodic only: Yes / No
- Periodicity threshold: 0-100%

---

## Conclusion

**Periodicity detection is a powerful enhancement** that:

1. ✅ Reduces false positives (preserves musical transients)
2. ✅ Validates detection (confirms clicks are real)
3. ✅ Enables smarter processing (adjust quality by confidence)
4. ✅ Provides user feedback (detected RPM)
5. ✅ Minimal computational cost

**Highly recommended** for implementation in production version.

---

## References

1. Godsill, S., & Rayner, P. (1998). *Digital Audio Restoration*. Springer. (Chapter 3: Click Detection)

2. Vaseghi, S.V., & Rayner, P.J.W. (1990). "Detection and suppression of impulsive noise in speech communication systems." *IEE Proceedings I*, 137(1), 38-46.

3. Kauppinen, I. (2002). "Methods for detecting impulsive noise in speech and audio signals." *14th International Conference on Digital Signal Processing*.

4. Esquef, P.A.A., et al. (2014). "Interpolation of long gaps in audio signals using the warped Burg method." *Audio Engineering Society Convention 137*.

5. Keiler, F., & Marchand, S. (2006). "Survey on extraction of sinusoids in stationary sounds." *Proc. Digital Audio Effects Conference*.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Author**: Claude (Anthropic)
