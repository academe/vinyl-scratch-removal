# Project Architecture

This document describes the architecture of the vinyl scratch removal project.

## Overview

The project uses a **core library + wrappers** architecture:

```
┌──────────────────────────────────────┐
│  Core Library (Cython)                │
│  - Click detection                    │
│  - AR interpolation                   │
│  - Cubic spline interpolation         │
└──────────────────────────────────────┘
              │
    ┌─────────┼─────────┬──────────┐
    │         │         │          │
┌───▼───┐ ┌──▼──┐ ┌────▼─────┐ ┌──▼────┐
│ LV2   │ │ CLI │ │ Nyquist  │ │Python │
│Plugin │ │Tool │ │ Plugin   │ │ Tool  │
└───────┘ └─────┘ └──────────┘ └───────┘
```

## Directory Structure

```
vinyl-scratch-removal/
├── core/                           # Cython core library
│   ├── detection.pyx              # Click detection (Cython)
│   ├── interpolation.pyx          # Interpolation algorithms (Cython)
│   ├── vinyl_core.pyx             # Main processor (Cython)
│   ├── vinyl_core.h               # C API header
│   └── setup.py                   # Build script
│
├── lv2/                            # LV2 plugin for DAWs
│   ├── vinyl_scratch_removal.c    # Plugin code
│   ├── vinyl_scratch_removal.ttl  # Plugin description
│   ├── manifest.ttl               # LV2 manifest
│   └── Makefile                   # Build system
│
├── cli/                            # Command-line tool
│   ├── vinyl                      # Executable wrapper
│   └── vinyl_cli.py               # CLI implementation
│
├── test/                           # Test suite
│   └── test_core.py               # Core library tests
│
├── docs/                           # Documentation
│   ├── RESEARCH_FINDINGS.md       # Algorithm research
│   ├── NYQUIST_PROGRAMMING_GUIDE.md
│   ├── IMPLEMENTATION_NOTES.md
│   ├── ALTERNATIVE_IMPLEMENTATIONS.md
│   └── ARCHITECTURE_AND_PORTABILITY.md
│
├── vinyl-scratch-removal.ny        # Original Nyquist plugin
├── vinyl_scratch_removal.py        # Original Python tool
│
├── BUILD.md                        # Build instructions
├── ARCHITECTURE.md                 # This file
├── build.sh                        # Quick build script
└── README.md                       # Main documentation
```

## Components

### Core Library (Cython)

**Location**: `core/`

**Purpose**: High-performance vinyl scratch removal algorithms

**Language**: Cython (Python with C types)

**Files**:
- `detection.pyx` - Click detection using second derivative analysis
- `interpolation.pyx` - AR and cubic spline interpolation
- `vinyl_core.pyx` - Main processor class
- `vinyl_core.h` - C API for integration with C/C++ code
- `setup.py` - Build configuration

**Build**:
```bash
cd core
python3 setup.py build_ext --inplace
```

**Outputs**:
- `detection.so` (or `.pyd` on Windows)
- `interpolation.so`
- `vinyl_core.so`

**Performance**: Near-C speed with type annotations

---

### LV2 Plugin

**Location**: `lv2/`

**Purpose**: Plugin for Audacity, Ardour, and other LV2-compatible DAWs

**Language**: C

**Files**:
- `vinyl_scratch_removal.c` - Plugin implementation
- `manifest.ttl` - LV2 manifest (required)
- `vinyl_scratch_removal.ttl` - Plugin description (parameters, etc.)
- `Makefile` - Build system

**Build**:
```bash
cd lv2
make
```

**Output**: `vinyl_scratch_removal.lv2/` bundle

**Install**:
```bash
make install PREFIX=$HOME/.local
```

**Current Implementation**:
- Uses simplified algorithm (similar to Nyquist plugin)
- Can be extended to call Cython core for better quality

---

### CLI Tool

**Location**: `cli/`

**Purpose**: Command-line audio processing

**Language**: Python

**Files**:
- `vinyl` - Executable wrapper script
- `vinyl_cli.py` - Main CLI implementation

**Usage**:
```bash
./cli/vinyl input.wav output.wav --threshold 3.0
```

**Dependencies**: Core library must be built first

---

### Test Suite

**Location**: `test/`

**Purpose**: Automated testing of core library

**Files**:
- `test_core.py` - Unit tests for detection and interpolation

**Run**:
```bash
./test/test_core.py
```

---

## Build Process

### 1. Core Library

**Technology**: Cython compiles `.pyx` files to C, then to shared libraries

**Steps**:
1. Cython compiles `.pyx` → `.c`
2. GCC compiles `.c` → `.so`
3. Python can `import vinyl_core`

**Advantages**:
- Python-like syntax
- C-level performance
- Can use NumPy/SciPy

---

### 2. LV2 Plugin

**Technology**: Standard C compilation

**Steps**:
1. GCC compiles `.c` → `.so`
2. Copies `.ttl` files to bundle
3. Bundle can be installed to LV2 directory

**Advantages**:
- Standard plugin format
- Works in any LV2 host
- No dependencies

---

### 3. CLI Tool

**Technology**: Python script

**Steps**:
1. No compilation needed
2. Uses core library (must be built)

**Advantages**:
- Easy to modify
- Full Python ecosystem
- Fast development

---

## Algorithm Flow

### Detection Phase

```
Audio Input
    │
    ├─> Calculate second derivative
    │
    ├─> Calculate local RMS
    │
    ├─> Adaptive threshold = RMS × sensitivity
    │
    ├─> Find samples exceeding threshold
    │
    ├─> Group into click regions
    │
    └─> Validate width and amplitude
         │
         └─> List of (start, end) click positions
```

### Interpolation Phase

```
For each detected click:
    │
    ├─> Extract context samples (before/after)
    │
    ├─> Try AR interpolation:
    │     ├─> Calculate autocorrelation
    │     ├─> Solve Yule-Walker equations
    │     ├─> Predict missing samples
    │     └─> Success? → Use AR result
    │
    ├─> Fallback to cubic spline if AR fails
    │
    ├─> Apply blend window
    │
    └─> Replace click samples
```

---

## Data Flow

```
User Input (WAV file)
    │
    ├──> [Core Library]
    │       ├─> Load audio (NumPy array)
    │       ├─> Detect clicks (detection.pyx)
    │       ├─> Interpolate clicks (interpolation.pyx)
    │       └─> Return processed audio
    │
    ├──> [CLI Tool]
    │       ├─> Parse arguments
    │       ├─> Call core library
    │       └─> Save output
    │
    ├──> [LV2 Plugin]
    │       ├─> Receive audio buffer
    │       ├─> Process in-place
    │       └─> Return to host
    │
    └──> [Nyquist Plugin]
            ├─> Frequency-domain processing
            └─> Return processed signal
```

---

## Why Cython?

**Advantages**:
1. ✅ Python-like syntax (easy for Python developers)
2. ✅ C-level performance (with type annotations)
3. ✅ Can use NumPy/SciPy (already optimized)
4. ✅ Gradual optimization (add types where needed)
5. ✅ No steep learning curve

**Compared to pure C++**:
- Easier to write and maintain
- Can prototype in pure Python first
- Still very fast (10-100x faster than Python)

**Compared to pure Python**:
- Much faster (C speed with types)
- Can release GIL for parallel processing
- Compiled to native code

---

## Extension Points

### Adding New Detection Methods

1. Edit `core/detection.pyx`
2. Add new function
3. Rebuild: `cd core && python3 setup.py build_ext --inplace`

### Adding New Interpolation Methods

1. Edit `core/interpolation.pyx`
2. Add new function
3. Update `vinyl_core.pyx` to use it
4. Rebuild

### Adding New Parameters

1. Update `VinylConfig` in `vinyl_core.h`
2. Update `VinylProcessor` in `vinyl_core.pyx`
3. Update LV2 TTL file
4. Update CLI argument parser

---

## Performance Characteristics

### Core Library (Cython)

- **Detection**: O(n) where n = number of samples
- **AR Interpolation**: O(k × p²) where k = number of clicks, p = AR order
- **Cubic Interpolation**: O(k) where k = number of clicks
- **Memory**: O(n) for audio buffer

### Typical Processing Time

**Test system**: 2.5 GHz CPU, 44.1 kHz audio

- **1 minute of audio**: ~2-5 seconds
- **10 minute album**: ~20-50 seconds
- **With AR order 20**: Baseline
- **With AR order 30**: +30% time
- **With AR order 50**: +100% time

---

## Dependencies

### Build Dependencies

- Python 3.7+
- Cython
- NumPy
- GCC or compatible C compiler
- make (for LV2 plugin)

### Runtime Dependencies

**Core library**:
- Python 3.7+
- NumPy
- SciPy
- soundfile

**LV2 plugin**:
- None (standalone binary)

**CLI tool**:
- Core library + its dependencies

---

## Future Enhancements

### Short-term

- [ ] Improve LV2 plugin to call Cython core
- [ ] Add LADSPA plugin variant
- [ ] Create Python wheel for easy installation
- [ ] Add more test coverage

### Medium-term

- [ ] Optimize hot paths in Cython
- [ ] Add multi-threading support
- [ ] Create GUI wrapper
- [ ] Support more audio formats

### Long-term

- [ ] Port to pure C++ for zero dependencies
- [ ] Add deep learning option
- [ ] Create VST3 plugin
- [ ] Real-time processing mode

---

## References

- **Cython Documentation**: https://cython.readthedocs.io/
- **LV2 Specification**: http://lv2plug.in/
- **NumPy C API**: https://numpy.org/doc/stable/reference/c-api/
- **Build Instructions**: See BUILD.md
- **Algorithm Details**: See docs/RESEARCH_FINDINGS.md

---

**Last Updated**: 2025-11-08
**Version**: 1.0.0
