# Research Findings: Vinyl Scratch Removal Algorithms

This document contains comprehensive research findings on digital signal processing techniques for vinyl record click, pop, and scratch removal.

## Table of Contents

1. [Introduction](#introduction)
2. [Types of Vinyl Artifacts](#types-of-vinyl-artifacts)
3. [Detection Algorithms](#detection-algorithms)
4. [Interpolation Methods](#interpolation-methods)
5. [Published Research](#published-research)
6. [Commercial Implementations](#commercial-implementations)
7. [Implementation Considerations](#implementation-considerations)

---

## Introduction

Vinyl record digitization suffers from impulsive noise caused by:
- Surface scratches
- Dust and debris
- Manufacturing defects
- Stylus damage
- Wear and tear

These artifacts manifest as clicks (short duration, <2ms) and pops (longer duration, 2-10ms) that are highly audible and disruptive to listening.

---

## Types of Vinyl Artifacts

### 1. Clicks

**Characteristics**:
- Very short duration (0.1-2ms)
- High amplitude relative to signal
- Sharp onset and offset
- Broadband frequency content
- Typically affects 1-50 samples at 44.1kHz

**Cause**: Small scratches, dust particles, static discharge

**Detection approach**: First and second derivative analysis

### 2. Pops

**Characteristics**:
- Medium duration (2-10ms)
- Very high amplitude
- May affect both channels in stereo
- Impulsive character

**Cause**: Larger scratches, deep gouges, debris

**Detection approach**: Amplitude threshold with width analysis

### 3. Crackle

**Characteristics**:
- Continuous low-level clicks
- Random distribution
- Lower amplitude than individual clicks
- High-frequency content

**Cause**: General surface wear, groove damage

**Detection approach**: Statistical analysis of high-frequency energy

### 4. Thumps

**Characteristics**:
- Low-frequency transients
- Longer duration (10-50ms)
- May have ringing/resonance

**Cause**: Warped records, physical shocks

**Detection approach**: Low-frequency band analysis

---

## Detection Algorithms

### 1. Threshold-Based Detection

**Simple Threshold**:
```
if abs(sample[n]) > threshold:
    potential_click = True
```

**Problems**:
- Fixed threshold doesn't adapt to varying signal levels
- False positives in loud passages
- Misses clicks in quiet passages

**Solution**: Adaptive thresholding

---

### 2. Adaptive Threshold Detection

**Algorithm**:
```
local_rms = RMS(signal[n-window:n+window])
threshold = local_rms * sensitivity_factor
if abs(signal[n]) > threshold:
    potential_click = True
```

**Advantages**:
- Adapts to local signal level
- Reduces false positives
- Handles dynamic range variations

**Parameters**:
- Window size: 10-20ms typical
- Sensitivity factor: 2.0-5.0 (standard deviations above RMS)

**Research source**: Godsill & Rayner (1998) - "Digital Audio Restoration"

---

### 3. Derivative-Based Detection

**First Derivative (Slope)**:
```
diff1[n] = signal[n] - signal[n-1]
```

Clicks show as rapid changes in slope.

**Second Derivative (Curvature)**:
```
diff2[n] = diff1[n] - diff1[n-1]
         = signal[n] - 2*signal[n-1] + signal[n-2]
```

Clicks show as high curvature values.

**Detection**:
```
if abs(diff2[n]) > threshold * local_rms:
    potential_click = True
```

**Advantages**:
- More sensitive to transients
- Less affected by signal amplitude
- Better discrimination of clicks vs music

**Research source**: Esquef et al. (2004) - "Detection of clicks using warped linear prediction"

---

### 4. Statistical Detection

**Z-score Method**:
```
mean = average(signal[window])
std_dev = standard_deviation(signal[window])
z_score[n] = (signal[n] - mean) / std_dev

if abs(z_score[n]) > threshold:
    outlier detected (potential click)
```

**Median Absolute Deviation (MAD)**:
```
median = median(signal[window])
mad = median(abs(signal - median))
threshold = median + k * mad

if abs(signal[n]) > threshold:
    potential_click = True
```

MAD is more robust to outliers than standard deviation.

---

### 5. Spectral Detection

**Approach**: Clicks have broadband frequency content unlike music.

**Algorithm**:
```
1. Compute STFT (Short-Time Fourier Transform)
2. Analyze spectral flatness
3. High spectral flatness = likely click
```

**Spectral Flatness**:
```
geometric_mean = (product of all magnitudes)^(1/N)
arithmetic_mean = sum of all magnitudes / N
flatness = geometric_mean / arithmetic_mean
```

Clicks ≈ 1.0 (flat, noise-like)
Music < 0.5 (harmonic structure)

**Limitation**: Computationally expensive, not ideal for Nyquist

---

### 6. Audacity's Algorithm

**Based on research and source code examination**:

```
Window size: 8192 samples
Separation: 2049 samples

For each window:
1. Find local maximum amplitude
2. Calculate local average
3. If max/average > threshold AND width < max_width:
   - Mark as click
4. Interpolate marked region
```

**Parameters**:
- Threshold: 0-900 (maps to ratio)
- Max spike width: 0-40 samples

**Source**: `src/effects/builtin/clickremoval/ClickRemovalBase.cpp`

---

## Interpolation Methods

Once a click is detected, the corrupted samples must be replaced.

### 1. Linear Interpolation

**Algorithm**:
```
for i in range(start, end):
    t = (i - start) / (end - start)
    signal[i] = (1-t) * signal[start-1] + t * signal[end]
```

**Advantages**:
- Simple, fast
- Always stable

**Disadvantages**:
- Audible for clicks > 5 samples
- Creates discontinuity in first derivative
- Unnatural sound

**Use case**: Very short clicks only (1-3 samples)

---

### 2. Cubic Hermite Spline Interpolation

**Algorithm**:
```
Given points P0, P1, P2, P3 around the click region:

1. Calculate tangents:
   m0 = (P1 - P0) / 2
   m3 = (P3 - P2) / 2

2. For each sample t in [0,1]:
   h00 = 2t³ - 3t² + 1
   h10 = t³ - 2t² + t
   h01 = -2t³ + 3t²
   h11 = t³ - t²

   interpolated = h00*P1 + h10*m0 + h01*P2 + h11*m3
```

**Advantages**:
- Smooth interpolation
- Continuous first derivative
- Natural-sounding
- Good for clicks up to 20-30 samples

**Disadvantages**:
- More complex than linear
- Requires context samples

**Research source**: Audio interpolation standard practice

---

### 3. Autoregressive (AR) Linear Prediction

**Concept**: Model the audio signal as a linear combination of past samples.

**AR Model**:
```
x[n] = a₁*x[n-1] + a₂*x[n-2] + ... + aₚ*x[n-p] + e[n]
```

Where:
- x[n] = current sample
- aᵢ = AR coefficients
- p = model order (typically 10-50)
- e[n] = prediction error (white noise)

**Algorithm**:

1. **Estimate AR coefficients** from clean samples before/after click:
   ```
   Solve Yule-Walker equations:
   R * a = r

   Where:
   R = autocorrelation matrix
   r = autocorrelation vector
   a = AR coefficients
   ```

2. **Forward prediction** from samples before click:
   ```
   for i in range(click_start, click_end):
       x[i] = sum(a[k] * x[i-k] for k in range(1, p+1))
   ```

3. **Backward prediction** from samples after click

4. **Blend** forward and backward predictions:
   ```
   weight = (i - start) / (end - start)
   x[i] = (1-weight) * forward[i] + weight * backward[i]
   ```

**Advantages**:
- Very natural sound
- Excellent for tonal music
- Preserves harmonic structure
- Can handle longer gaps (up to 100ms)

**Disadvantages**:
- Computationally expensive
- Requires matrix operations
- May be unstable for high orders
- Difficult to implement in Nyquist

**Research sources**:
- Godsill & Rayner (1998) - "Digital Audio Restoration: A Statistical Model Based Approach"
- Lagrange et al. (2020) - "Restoration Based on High Order Sparse Linear Prediction"
- Etter (1996) - "Restoration of a discrete-time signal segment by interpolation"

**Implementation notes**:
```python
# Python/NumPy implementation (reference)
from scipy.linalg import solve_toeplitz

# Calculate autocorrelation
r = np.correlate(clean_signal, clean_signal, mode='full')
r = r[len(r)//2:][:p+1]

# Solve Yule-Walker equations
ar_coeffs = solve_toeplitz(r[:p], r[1:p+1])

# Predict missing samples
for i in range(gap_length):
    prediction = np.dot(ar_coeffs, previous_samples[::-1])
```

---

### 4. Polynomial Interpolation

**Lagrange Polynomial**:
```
Given n points (x₀,y₀), ..., (xₙ,yₙ):

L(x) = Σ yᵢ * lᵢ(x)

where lᵢ(x) = Π (x - xⱼ)/(xᵢ - xⱼ) for j≠i
             j=0
```

**Advantages**:
- Passes through all control points
- Smooth curve

**Disadvantages**:
- Runge's phenomenon (oscillations) for high degrees
- Unstable for audio interpolation

**Conclusion**: Not recommended for audio restoration

---

### 5. Sinc Interpolation (Band-Limited)

**Theory**: Perfect reconstruction for band-limited signals per Nyquist-Shannon theorem.

**Algorithm**:
```
x(t) = Σ x[n] * sinc((t - n*T) / T)
       n

where sinc(x) = sin(πx) / (πx)
```

**Advantages**:
- Theoretically perfect for band-limited signals
- Preserves frequency content

**Disadvantages**:
- Infinite support (must truncate)
- Assumes click-free samples are perfect
- Edge effects
- Computationally expensive

**Use case**: Sample rate conversion, not click removal

---

## Published Research

### Key Papers

#### 1. Godsill & Rayner (1998)
**Title**: "Digital Audio Restoration: A Statistical Model Based Approach"

**Key contributions**:
- Bayesian framework for click detection
- AR modeling for interpolation
- Sequential algorithm for real-time processing
- EM algorithm for parameter estimation

**Algorithm summary**:
```
1. Model clean signal as AR process
2. Model clicks as additive noise bursts
3. Use Kalman filter for joint detection/restoration
4. Iterate to refine estimates
```

**Practical impact**: Foundation for modern audio restoration

---

#### 2. Lagrange et al. (2020)
**Title**: "Restoration of Click Degraded Speech and Music Based on High Order Sparse Linear Prediction"

**Key contributions**:
- Sparse linear prediction reduces computation
- High-order models (p=50-100) for better quality
- Specific optimizations for music vs speech

**Algorithm improvement**:
- Traditional AR: O(p²) complexity
- Sparse AR: O(p*k) where k << p

**Results**:
- Better quality than cubic spline
- Faster than full AR prediction

---

#### 3. Esquef et al. (2004)
**Title**: "Detection of clicks in audio signals using warped linear prediction"

**Key contributions**:
- Warped frequency scale matches auditory perception
- Better discrimination of clicks vs music
- Reduced false positives

**Warped LP**:
```
Regular LP: Uniform frequency resolution
Warped LP: Higher resolution at low frequencies
          (matches human hearing)
```

---

#### 4. Vaseghi & Rayner (1990)
**Title**: "Detection and suppression of impulsive noise in speech communication systems"

**Key contributions**:
- Two-pass detection (forward/backward)
- Extended Kalman Filter approach
- Handles both detection and interpolation jointly

---

#### 5. Recent: Diffusion Models (2024)
**Title**: "Diffusion Models for Audio Restoration" (arXiv:2402.09821)

**Key contributions**:
- Deep learning approach using diffusion models
- Learns restoration from examples
- Can handle complex degradations

**Limitation**:
- Requires GPU, training data
- Not feasible for Nyquist plugin
- Future research direction

---

## Commercial Implementations

### 1. iZotope RX

**Approach**:
- Spectral editing with visual interface
- Machine learning for detection
- Multiple interpolation methods
- Real-time preview

**Features**:
- De-click module
- Spectral repair
- Adaptive detection
- Manual drawing mode

**Algorithm (inferred)**:
- STFT-based detection
- Spectral interpolation
- Deep learning enhancement (recent versions)

---

### 2. Cedar Audio

**Approach**:
- Professional hardware/software
- Real-time processing
- Multi-band detection

**Features**:
- Separate click and crackle removal
- Adaptive thresholding
- Preserves transients

**Algorithm (inferred)**:
- Multi-band filtering
- Adaptive attenuation (reduces rather than removes)
- Linear-phase filtering

---

### 3. Audacity Built-in

**Approach**:
- Simple threshold-based detection
- Basic interpolation

**Algorithm** (from source code):
```cpp
// Simplified from ClickRemovalBase.cpp
const int windowSize = 8192;
const int sep = 2049;

for each window:
    find max amplitude
    calculate average
    if (max/avg > threshold) && (width < maxWidth):
        mark as click
        interpolate
```

**Interpolation**: Linear between edges

**Limitations**:
- Fixed window size
- Simple interpolation
- No AR prediction
- Can damage transients

---

### 4. ClickRepair (Brian Davies)

**Approach**:
- Specialized for vinyl restoration
- Multiple detection algorithms
- High-quality interpolation

**Features**:
- Automatic and manual modes
- Adjustable sensitivity
- Preview capability

**Algorithm (inferred from documentation)**:
- Derivative-based detection
- Cubic spline interpolation
- Adaptive thresholding

---

## Implementation Considerations

### For Nyquist (LISP-based)

#### Limitations

1. **No sample-level access**: Nyquist processes audio as continuous signals, not individual samples

2. **No array manipulation**: Can't easily implement AR prediction (requires matrix operations)

3. **Limited control structures**: Difficult to implement complex algorithms

4. **Memory constraints**: Large buffers may cause issues

5. **No FFT access**: Can't do spectral editing

#### Feasible Approaches in Nyquist

1. **Frequency-domain separation**:
   ```lisp
   (setf low-freq (lowpass8 signal 2000))
   (setf high-freq (highpass8 signal 2000))
   ; Clicks are in high-freq component
   ```

2. **Envelope-based detection**:
   ```lisp
   (setf envelope (snd-avg signal window-size))
   ```

3. **Adaptive attenuation** (not removal):
   ```lisp
   ; Reduce amplitude rather than interpolate
   (mult signal (min 1.0 (/ threshold envelope)))
   ```

4. **Built-in functions**:
   - `lowpass8`, `highpass8`: Filtering
   - `snd-avg`: Moving average
   - `snd-abs`: Absolute value
   - `mult`, `sum`: Signal combination

#### Recommended Nyquist Strategy

**Two-stage approach**:

1. **Separate signal components**:
   - Low frequencies (music): < 2kHz
   - High frequencies (transients + clicks): > 2kHz

2. **Process high-frequency component**:
   - Calculate envelope
   - Create adaptive threshold
   - Attenuate (not remove) clicks
   - Reduce gain by 50-70% on detected clicks

3. **Recombine**:
   - Add processed high-freq back to low-freq

**Advantages**:
- Works within Nyquist constraints
- No sample-level manipulation needed
- Preserves most musical content
- Computationally efficient

**Code structure**:
```lisp
(setf low (lowpass8 s 2000))
(setf high (highpass8 s 2000))
(setf high-rms (local-rms high))
(setf threshold (mult high-rms sensitivity))
(setf attenuation (compute-attenuation high threshold))
(setf high-processed (mult high attenuation))
(sum low high-processed)
```

---

### For Python/SciPy Implementation

Python allows full implementation of advanced algorithms:

1. **AR Linear Prediction**:
   ```python
   from scipy.linalg import solve_toeplitz
   ar_coeffs = solve_toeplitz(R, r)
   ```

2. **Cubic Spline**:
   ```python
   from scipy.interpolate import CubicSpline
   cs = CubicSpline(x_context, y_context)
   interpolated = cs(x_missing)
   ```

3. **Sample-level control**:
   ```python
   for i, (start, end) in enumerate(clicks):
       audio[start:end] = interpolate_ar(audio, start, end)
   ```

4. **Advanced detection**:
   ```python
   diff2 = np.diff(np.diff(audio))
   local_rms = sliding_window_rms(audio, window_size)
   threshold = local_rms * sensitivity
   clicks = find_regions(diff2 > threshold)
   ```

---

## Algorithm Comparison

| Method | Quality | Speed | Complexity | Nyquist? |
|--------|---------|-------|------------|----------|
| Linear Interpolation | Low | Fast | Simple | Yes |
| Cubic Spline | Good | Fast | Medium | No* |
| AR Prediction | Excellent | Slow | High | No |
| Spectral Repair | Excellent | Medium | High | No |
| Attenuation | Fair | Fast | Simple | Yes |
| Deep Learning | Excellent | Slow** | Very High | No |

*Possible with workarounds
**Fast with GPU

---

## Best Practices

### 1. Detection

- **Use adaptive thresholds**: Signal level varies throughout recording
- **Multi-pass processing**: First pass for obvious clicks, second for subtle ones
- **Validate width**: Ignore very wide "clicks" (likely music)
- **Check both channels**: Stereo clicks may appear in one or both

### 2. Interpolation

- **Match method to gap size**:
  - 1-3 samples: Linear
  - 4-20 samples: Cubic spline
  - 20+ samples: AR prediction

- **Use sufficient context**: AR prediction needs 2-4x model order samples on each side

- **Blend edges**: Avoid discontinuities with windowing

### 3. Parameter Tuning

- **Start conservative**: High threshold, narrow width
- **Increase gradually**: If clicks remain
- **Monitor for artifacts**: "Watery" sound = over-processing
- **Genre-specific**: Classical needs more care than rock

### 4. Validation

- **A/B testing**: Compare before/after
- **Visual inspection**: Waveform shows if clicks removed
- **Spectrum analysis**: Check for unnatural frequency content
- **Listen carefully**: Use good headphones

---

## Future Directions

### 1. Machine Learning Approaches

**Supervised Learning**:
- Train on click/no-click examples
- Learn optimal detection parameters
- Neural network for classification

**Diffusion Models**:
- Recent research (2024) shows promise
- Generate clean audio conditioned on noisy input
- Requires training data and GPU

### 2. Real-time Processing

**Challenges**:
- AR prediction requires future samples
- Latency vs quality tradeoff

**Solutions**:
- Look-ahead buffer
- Simplified algorithms for RT
- Hardware acceleration

### 3. Adaptive Algorithms

**Self-tuning**:
- Analyze recording to estimate optimal parameters
- Learn vinyl characteristics
- Adjust during playback

### 4. Perceptual Modeling

**Psychoacoustic masking**:
- Clicks in loud passages less audible
- Adjust detection based on masking threshold
- Frequency-dependent sensitivity

---

## References

1. Godsill, S., & Rayner, P. (1998). *Digital Audio Restoration: A Statistical Model Based Approach*. Springer-Verlag.

2. Vaseghi, S. V., & Rayner, P. J. W. (1990). "Detection and suppression of impulsive noise in speech communication systems." *IEE Proceedings I*, 137(1), 38-46.

3. Esquef, P. A., Biscainho, L. W., Diniz, P. S., & Freeland, F. P. (2004). "Detection of clicks in audio signals using warped linear prediction." *IEEE Workshop on Applications of Signal Processing to Audio and Acoustics*.

4. Lagrange, M., et al. (2020). "Restoration of Click Degraded Speech and Music Based on High Order Sparse Linear Prediction." *ResearchGate*.

5. Etter, W. (1996). "Restoration of a discrete-time signal segment by interpolation based on the left-sided and right-sided autoregressive parameters." *IEEE Transactions on Signal Processing*, 44(5), 1124-1135.

6. Kauppinen, I. (2002). "Methods for detecting impulsive noise in speech and audio signals." *Proceedings of the 14th International Conference on Digital Signal Processing*.

7. Niedźwiecki, M., & Cisowski, K. (2001). "Smart copying - a new approach to reconstruction of audio signals." *IEEE Transactions on Signal Processing*, 49(10), 2272-2282.

8. Abel, J. S., & Smith, J. O. (2005). "Robust design of very high-order allpass dispersion filters." *Proceedings of DAFx-05*.

9. Recent (2024): "Diffusion Models for Audio Restoration." *arXiv:2402.09821*.

10. Columbia University EE Course: "Audio Click Removal Using Linear Prediction" - https://www.ee.columbia.edu/~dpwe/classes/e4810-2004-09/

---

## Conclusion

Vinyl scratch removal is a well-researched problem with solutions ranging from simple threshold detection to advanced machine learning. The choice of algorithm depends on:

- **Target quality**: Casual listening vs archival restoration
- **Computational resources**: Real-time vs offline
- **Implementation platform**: Nyquist limitations vs Python flexibility
- **User expertise**: Automatic vs manual tuning

For Nyquist implementation, frequency-domain separation with adaptive attenuation provides the best balance of quality and feasibility within platform constraints. For highest quality, Python implementation with AR prediction is recommended.

---

*Last updated: 2025-11-07*
*Research compiled by: Claude (Anthropic)*
