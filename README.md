# Vinyl Scratch Removal for Audacity

Advanced vinyl record scratch, click, and pop removal tools using research-backed digital signal processing techniques.

## Overview

This project provides two implementations for removing scratches, clicks, and pops from vinyl record digitizations:

1. **Nyquist Plugin** - Easy-to-use plugin for Audacity
2. **Python Tool** - Advanced standalone processor with state-of-the-art algorithms

## Features

### Research-Backed Techniques

Both implementations use algorithms based on published research:

- **Autoregressive (AR) Linear Prediction** - For natural-sounding sample interpolation
- **Cubic Hermite Spline Interpolation** - Smooth reconstruction with continuous derivatives
- **Adaptive Threshold Detection** - Uses local RMS analysis for robust click detection
- **Multi-pass Processing** - Handles different types of artifacts
- **Slope Analysis** - Detects rapid transients characteristic of clicks

### Key Capabilities

- Removes clicks, pops, and scratches from vinyl recordings
- Adaptive detection adjusts to local audio characteristics
- Preserves musical content while removing artifacts
- Multiple detection modes (conservative, standard, aggressive)
- Processes both mono and stereo audio
- Real-time preview in Audacity (Nyquist plugin)

## Installation

### Nyquist Plugin (Audacity)

1. **Download the plugin**:
   - Copy `vinyl-scratch-removal.ny` to your Audacity plugins folder:
     - **Windows**: `C:\Program Files\Audacity\Plug-Ins\`
     - **macOS**: `~/Library/Application Support/audacity/Plug-Ins/`
     - **Linux**: `~/.audacity-data/Plug-Ins/` or `/usr/share/audacity/plug-ins/`

2. **Enable in Audacity**:
   - Open Audacity
   - Go to `Tools > Plugin Manager`
   - Find "Vinyl Scratch Removal" and enable it
   - Click `OK`

3. **Use the plugin**:
   - Select audio in Audacity
   - Go to `Effect > Vinyl Scratch Removal`
   - Adjust parameters (see Usage section)
   - Click `Preview` to test, then `Apply`

### Python Tool

1. **Install Python 3.7+** (if not already installed)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install numpy scipy soundfile
   ```

## Usage

### Nyquist Plugin

**Parameters**:

- **Click Sensitivity (1-100)**: How aggressively to detect clicks
  - `1-10`: Very conservative, only obvious clicks
  - `20`: Default, balanced detection
  - `50-100`: Aggressive, may detect false positives

- **Maximum Click Width (ms)**: Maximum duration of clicks to detect
  - `0.1-2.0`: Short clicks only (default: 2.0ms)
  - `2.0-5.0`: Medium clicks and pops
  - `5.0-10.0`: Large pops and scratches

- **Detection Mode**:
  - `Clicks Only`: Conservative, single-pass
  - `Clicks + Crackle`: Standard, multi-pass (recommended)
  - `Aggressive`: Maximum detection, may affect audio

**Recommended Settings**:

- **Light cleaning**: Sensitivity=10, Width=2.0ms, Mode=Clicks Only
- **Standard restoration**: Sensitivity=20, Width=2.0ms, Mode=Clicks + Crackle
- **Heavy restoration**: Sensitivity=30, Width=5.0ms, Mode=Aggressive

### Python Tool

**Basic usage**:
```bash
python vinyl_scratch_removal.py input.wav output.wav
```

**With custom parameters**:
```bash
# Conservative (only obvious clicks)
python vinyl_scratch_removal.py input.wav output.wav --mode conservative --threshold 4.0

# Standard (recommended)
python vinyl_scratch_removal.py input.wav output.wav --mode standard --threshold 3.0

# Aggressive (maximum click removal)
python vinyl_scratch_removal.py input.wav output.wav --mode aggressive --threshold 2.0 --max-width 5.0
```

**Command-line options**:

```
--threshold FLOAT       Detection threshold in standard deviations
                        Lower = more sensitive (default: 3.0)
                        Range: 1.0 (very sensitive) to 5.0 (conservative)

--max-width FLOAT       Maximum click width in milliseconds
                        (default: 2.0)

--mode MODE             Detection mode:
                        - conservative: Only obvious clicks
                        - standard: Balanced (default)
                        - aggressive: Maximum detection

--ar-order INT          AR model order for interpolation
                        Higher = more accurate but slower
                        (default: 20, range: 10-50)
```

## How It Works

### Click Detection

1. **Local Statistics Analysis**
   - Calculates local RMS in sliding windows (10ms)
   - Computes first and second derivatives of the signal
   - Identifies rapid amplitude changes

2. **Adaptive Thresholding**
   - Threshold adapts to local signal level
   - Avoids false detections in loud passages
   - Preserves musical transients (drums, etc.)

3. **Validation**
   - Checks click width against maximum
   - Verifies amplitude is significantly higher than surroundings
   - Merges nearby clicks

### Interpolation

The tools use two interpolation methods:

1. **Autoregressive Linear Prediction** (Python only)
   - Models the audio signal using AR coefficients
   - Predicts missing samples based on context
   - More accurate for tonal content
   - Based on research: "Restoration of Click Degraded Speech and Music"

2. **Cubic Hermite Spline** (both implementations)
   - Uses cubic polynomials for smooth interpolation
   - Maintains continuous first derivatives
   - Natural-sounding reconstruction
   - Fast and robust

### Multi-pass Processing

- First pass: Detect and remove obvious clicks
- Second pass: Detect remaining artifacts with reduced threshold
- Prevents over-processing while catching all clicks

## Technical Background

This implementation is based on the following research:

1. **Godsill, S. & Rayner, P.** - "Digital Audio Restoration: A Statistical Model Based Approach"
   - Statistical methods for click detection
   - AR modeling for interpolation

2. **Lagrange, M., et al.** - "Restoration of Click Degraded Speech and Music Based on High Order Sparse Linear Prediction" (ResearchGate)
   - Sparse linear prediction techniques
   - High-order AR models

3. **Esquef, P.A.A., et al.** - "Detection of clicks in audio signals using warped linear prediction"
   - Warped linear prediction for detection
   - Adaptive threshold methods

4. **Recent advances** - "Diffusion Models for Audio Restoration" (2024)
   - Modern deep learning approaches
   - (Future implementation target)

## Comparison with Other Tools

### Audacity Built-in Click Removal

- **Built-in**: Simple threshold-based detection with basic interpolation
- **This plugin**: Adaptive detection with AR/cubic spline interpolation
- **Advantage**: Better preservation of musical content, more accurate click detection

### Commercial Tools (iZotope RX, etc.)

- **Commercial**: Advanced spectral editing, machine learning, visual interface
- **This plugin**: Free, open-source, research-backed algorithms
- **Advantage**: Free, customizable, transparent algorithms

## Limitations

1. **Processing Time**: AR interpolation is CPU-intensive for long files
2. **Nyquist Constraints**: Nyquist plugin uses simplified algorithms due to language limitations
3. **Not for All Noise**: Designed for impulsive clicks/pops, not continuous noise (hiss, hum)
4. **Manual Tuning**: May require parameter adjustment for different recordings

## Best Practices

### Before Processing

1. **Clean the record** - Physical cleaning reduces scratches
2. **Test a section** - Process a small section first to tune parameters
3. **Use preview** - In Audacity, always preview before applying
4. **Make a backup** - Save original before processing

### Parameter Tuning

1. **Start conservative** - Begin with low sensitivity
2. **Increase gradually** - If clicks remain, increase sensitivity
3. **Check for artifacts** - Listen for "watery" or "bubbling" sounds (over-processing)
4. **Adjust width** - Match maximum click width to your vinyl's condition

### Post-Processing

1. **Check edges** - Listen to track beginnings/ends for artifacts
2. **Compare with original** - A/B test to ensure quality
3. **Further processing** - Consider additional noise reduction if needed
4. **Normalize** - May want to normalize volume after processing

## Troubleshooting

### "Still hearing clicks"

- **Increase sensitivity**: Try threshold 30-50 (Nyquist) or 2.0-2.5 (Python)
- **Increase max width**: Try 5-10ms for larger pops
- **Use aggressive mode**: Switch to aggressive detection

### "Audio sounds damaged/watery"

- **Decrease sensitivity**: Try threshold 5-10 (Nyquist) or 4.0-5.0 (Python)
- **Use conservative mode**: Switch to conservative detection
- **Reduce max width**: Try 1.0-2.0ms

### "Plugin not appearing in Audacity"

1. Check plugin file location
2. Restart Audacity
3. Check Plugin Manager (Tools > Plugin Manager)
4. Enable plugin if disabled

### "Python script errors"

- **Check dependencies**: `pip install -r requirements.txt`
- **Check file format**: Use WAV files (MP3/FLAC may need conversion)
- **Check Python version**: Requires Python 3.7+

## Contributing

Contributions welcome! Areas for improvement:

- Deep learning-based detection and interpolation
- Real-time processing optimization
- GUI for Python tool
- Additional file format support
- LADSPA/LV2 plugin compilation

## License

MIT License - See LICENSE file for details

## Credits

- **Algorithm Research**: Based on published DSP research (see Technical Background)
- **Implementation**: Claude (Anthropic)
- **Testing**: Community contributions welcome

## References

1. Godsill, S., & Rayner, P. (1998). *Digital Audio Restoration: A Statistical Model Based Approach*. Springer.

2. Lagrange, M., et al. (2020). "Restoration of Click Degraded Speech and Music Based on High Order Sparse Linear Prediction." *ResearchGate*.

3. Esquef, P.A.A., et al. (2004). "Detection of clicks in audio signals using warped linear prediction." *IEEE*.

4. Columbia University DSP Course. "Audio Click Removal Using Linear Prediction."

5. Various authors (2024). "Diffusion Models for Audio Restoration." *arXiv:2402.09821*.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult Audacity forums for plugin-specific questions

---

**Note**: This is research software. While based on published algorithms, results may vary depending on your audio. Always keep original recordings as backup.
