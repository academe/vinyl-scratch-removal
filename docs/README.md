# Technical Documentation

This directory contains comprehensive technical documentation for the Vinyl Scratch Removal project.

## Documentation Overview

### [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md)
**Comprehensive research on vinyl scratch removal algorithms**

Topics covered:
- Types of vinyl artifacts (clicks, pops, crackle, thumps)
- Detection algorithms (threshold-based, derivative-based, statistical, spectral)
- Interpolation methods (linear, cubic spline, AR prediction, sinc)
- Published research papers and key contributions
- Commercial implementations (iZotope RX, Cedar Audio, ClickRepair)
- Algorithm comparisons and best practices

**Who should read**: Anyone interested in the theoretical foundations and research background.

---

### [NYQUIST_PROGRAMMING_GUIDE.md](NYQUIST_PROGRAMMING_GUIDE.md)
**Complete guide to Nyquist programming for Audacity plugins**

Topics covered:
- Introduction to Nyquist language and philosophy
- Plugin structure and headers
- Audio processing concepts (signal algebra, time vs samples)
- Available functions (filtering, analysis, generation)
- Limitations and workarounds
- Best practices and design patterns
- Example implementations
- Debugging tips

**Who should read**: Developers creating or modifying Nyquist plugins.

---

### [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)
**Detailed explanation of the vinyl scratch removal plugin implementation**

Topics covered:
- Design goals and constraints
- Algorithm selection rationale
- Plugin architecture and data flow
- Implementation details for each component
- Limitations and trade-offs
- Testing methodology
- Future improvements
- Nyquist vs Python implementation comparison

**Who should read**: Developers wanting to understand or modify this specific plugin.

---

### [ALTERNATIVE_IMPLEMENTATIONS.md](ALTERNATIVE_IMPLEMENTATIONS.md)
**Can the high-quality Python algorithm be built into a Nyquist plugin?**

Topics covered:
- Why AR interpolation cannot work in Nyquist (fundamental constraints)
- Detailed explanation of what's needed vs what Nyquist provides
- Alternative plugin formats (LADSPA, LV2, VST, native C++)
- Hybrid approaches (external calls, scripting integration)
- Implementation roadmap for each approach
- Effort vs benefit analysis
- Recommended path forward

**Who should read**: Anyone wondering why there are two implementations, or developers considering creating a compiled plugin.

**Key takeaway**: Nyquist's signal-based paradigm makes AR interpolation impossible. Best alternatives are LV2 plugin or native Audacity C++ effect.

---

### [ARCHITECTURE_AND_PORTABILITY.md](ARCHITECTURE_AND_PORTABILITY.md)
**How to create a portable core library with multiple wrappers (LV2, CLI, etc.)**

Topics covered:
- Core library + wrapper architecture (one implementation, many interfaces)
- Python compilation options (Cython, PyBind11, embedded interpreter)
- Detailed comparison: Cython vs C++ core
- Architecture designs (pure C++, Cython, hybrid)
- Practical migration path from Python to compiled library
- Build systems (CMake, setup.py)
- Step-by-step implementation roadmap

**Who should read**: Developers wanting to create LV2/LADSPA plugins, or anyone interested in portable library design.

**Key takeaway**: Core library + wrappers is the right approach. Python can be compiled via Cython (great for Python developers), or port to C++ for maximum portability.

---

## Quick Reference

### Algorithm Summary

**Nyquist Plugin Approach**:
1. Separate audio into low-frequency (<2kHz) and high-frequency (>2kHz) components
2. Calculate local RMS envelope of high-frequency component
3. Create adaptive threshold based on envelope and user sensitivity
4. Attenuate high-frequency component where it exceeds threshold
5. Recombine processed high-freq with unchanged low-freq
6. Multi-pass processing with decreasing threshold

**Python Tool Approach**:
1. Calculate second derivative of audio signal
2. Detect clicks using adaptive threshold based on local RMS
3. Validate detected clicks (width, amplitude criteria)
4. Interpolate using autoregressive linear prediction
5. Fall back to cubic Hermite spline if AR fails
6. Blend interpolated samples with windowing

### Key Research Papers

1. **Godsill & Rayner (1998)** - "Digital Audio Restoration: A Statistical Model Based Approach"
   - Foundation for modern audio restoration
   - AR modeling and Bayesian detection

2. **Lagrange et al. (2020)** - "Restoration Based on High Order Sparse Linear Prediction"
   - Sparse AR for computational efficiency
   - High-order models for quality

3. **Esquef et al. (2004)** - "Detection of clicks using warped linear prediction"
   - Perceptually-weighted detection
   - Reduced false positives

4. **Recent (2024)** - "Diffusion Models for Audio Restoration"
   - Deep learning approach
   - Future research direction

### Nyquist Limitations

**Cannot do**:
- Sample-level array access (`audio[i]`)
- Matrix operations (AR coefficient solving)
- FFT/spectral editing
- Complex loops over samples
- Machine learning

**Can do**:
- Signal-level operations
- Filtering (low-pass, high-pass, band-pass)
- Signal algebra (multiply, add, etc.)
- Envelope following
- Adaptive processing with signal-level threshold

### Algorithm Comparison

| Method | Quality | Speed | Nyquist? | Use Case |
|--------|---------|-------|----------|----------|
| Adaptive Attenuation | Good | Fast | ✓ Yes | Real-time, good enough |
| Linear Interpolation | Fair | Fast | ✗ No* | Very short clicks only |
| Cubic Spline | Good | Fast | ✗ No* | Short clicks (<20 samples) |
| AR Prediction | Excellent | Slow | ✗ No | Highest quality, offline |
| Spectral Repair | Excellent | Medium | ✗ No | Tonal music |
| Deep Learning | Excellent | Slow** | ✗ No | Research, future |

*Requires sample-level access
**Fast with GPU

---

## Development Workflow

### For Nyquist Plugin Development

1. **Prototype in Nyquist Prompt** (Tools > Nyquist Prompt in Audacity)
   ```lisp
   (lowpass2 s 1000)  ; Test snippets
   ```

2. **Create .ny file** with proper headers
   ```lisp
   ;nyquist plug-in
   ;version 4
   ;type process
   ;name "My Effect"
   ```

3. **Test with sample audio** in Audacity
   - Generate test tones
   - Add synthetic clicks
   - Measure results

4. **Refine parameters** based on testing

5. **Document** code and usage

### For Python Development

1. **Set up environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy scipy soundfile
   ```

2. **Implement algorithm** in Python
   - Use NumPy for sample arrays
   - SciPy for signal processing
   - Test with small audio snippets

3. **Validate** against research papers
   - Compare results with published methods
   - Measure SNR, click reduction

4. **Optimize** for performance
   - Profile code
   - Vectorize operations
   - Consider C extensions for critical loops

---

## Resources

### Audacity
- **Audacity Manual**: https://manual.audacityteam.org/
- **Nyquist Documentation**: https://manual.audacityteam.org/man/nyquist.html
- **Plugin Repository**: https://plugins.audacityteam.org/

### Nyquist Language
- **Nyquist Reference**: https://www.cs.cmu.edu/~rbd/doc/nyquist/
- **CMU Nyquist Page**: https://www.cs.cmu.edu/~music/nyquist/

### DSP Resources
- **"Digital Audio Restoration" book** by Godsill & Rayner (1998)
- **DSP Guide**: http://www.dspguide.com/
- **DAFX book**: "Digital Audio Effects" by Udo Zölzer

### Academic Papers
- See RESEARCH_FINDINGS.md References section for complete bibliography

### Communities
- **Audacity Forum**: https://forum.audacityteam.org/
- **DSP Stack Exchange**: https://dsp.stackexchange.com/
- **Music DSP**: https://www.musicdsp.org/

---

## File Structure

```
docs/
├── README.md (this file)
│   Overview of documentation
│
├── RESEARCH_FINDINGS.md
│   └── Complete research on vinyl scratch removal
│       - Detection algorithms
│       - Interpolation methods
│       - Published papers
│       - Commercial implementations
│
├── NYQUIST_PROGRAMMING_GUIDE.md
│   └── Nyquist language reference
│       - Language concepts
│       - Available functions
│       - Limitations and workarounds
│       - Examples and patterns
│
├── IMPLEMENTATION_NOTES.md
│   └── This plugin's implementation
│       - Design decisions
│       - Algorithm details
│       - Trade-offs
│       - Testing and future work
│
├── ALTERNATIVE_IMPLEMENTATIONS.md
│   └── Beyond Nyquist constraints
│       - Why AR interpolation can't work in Nyquist
│       - Alternative plugin formats (LADSPA, LV2, C++)
│       - Hybrid approaches
│       - Implementation roadmap
│
└── ARCHITECTURE_AND_PORTABILITY.md
    └── Portable library design
        - Core library + wrapper architecture
        - Compiling Python to library (Cython)
        - C++ porting strategy
        - Build systems and migration path
```

---

## Glossary

**AR (Autoregressive)**: Statistical model where current value depends on previous values. Used for prediction and interpolation.

**Click**: Short-duration (0.1-2ms) high-amplitude impulsive noise from vinyl surface defects.

**Crackle**: Continuous low-level clicking noise from general wear.

**Cubic Hermite Spline**: Interpolation method using cubic polynomials with continuous first derivatives.

**FFT (Fast Fourier Transform)**: Algorithm for computing frequency spectrum of audio.

**LISP**: Family of programming languages based on lambda calculus. Nyquist is a LISP dialect.

**Nyquist**: 1) Programming language for audio. 2) Harry Nyquist, electrical engineer. 3) Nyquist-Shannon sampling theorem.

**Pop**: Medium-duration (2-10ms) high-amplitude impulse from larger scratches.

**RMS (Root Mean Square)**: Measure of signal level/energy.

**Sample Rate**: Number of samples per second (e.g., 44100 Hz = 44,100 samples/second).

**STFT (Short-Time Fourier Transform)**: FFT applied to overlapping windows of audio.

**Threshold**: Level above which clicks are detected.

---

## Contributing to Documentation

### Adding New Information

1. **Research findings**: Add to RESEARCH_FINDINGS.md
   - Include proper citations
   - Explain algorithm clearly
   - Provide examples if possible

2. **Nyquist techniques**: Add to NYQUIST_PROGRAMMING_GUIDE.md
   - Include working code examples
   - Explain limitations
   - Show use cases

3. **Implementation details**: Add to IMPLEMENTATION_NOTES.md
   - Document design decisions
   - Explain trade-offs
   - Update based on testing

### Documentation Standards

- **Clear headings**: Use markdown hierarchy
- **Code examples**: Syntax-highlighted, well-commented
- **Citations**: Include sources for research claims
- **Cross-references**: Link between documents
- **Practical examples**: Show real use cases
- **Keep updated**: Revise when implementation changes

---

## Version History

### Version 1.0 (2025-11-07)
- Initial documentation
- Complete research compilation
- Nyquist programming guide
- Implementation notes

### Future Additions Planned
- Performance benchmarks
- User case studies
- Video tutorials
- Additional algorithm implementations

---

## Contact and Feedback

For questions, corrections, or additions to documentation:
- Open an issue on GitHub
- Contribute pull requests
- Share your vinyl restoration experiences

---

**Last Updated**: 2025-11-07
**Maintained By**: Claude (Anthropic)
**License**: MIT (see LICENSE file)
