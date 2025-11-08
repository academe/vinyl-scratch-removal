# Architecture and Portability Strategy

This document outlines a portable, maintainable architecture for implementing high-quality vinyl scratch removal across multiple platforms and interfaces.

## Table of Contents

1. [Core Library + Wrapper Architecture](#core-library--wrapper-architecture)
2. [Python as Core Implementation](#python-as-core-implementation)
3. [Compilation Options for Python](#compilation-options-for-python)
4. [Architecture Designs](#architecture-designs)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Recommended Approach](#recommended-approach)

---

## Core Library + Wrapper Architecture

### Concept

**One algorithm implementation, multiple interfaces:**

```
┌─────────────────────────────────────┐
│     Core Scratch Removal Library    │
│   (Algorithm implementation)         │
│   - Click detection                  │
│   - AR interpolation                 │
│   - Parameter handling               │
└─────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬─────────────┐
    │            │            │             │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐  ┌─────▼──────┐
│  LV2  │  │ LADSPA  │  │  CLI   │  │   Python   │
│Plugin │  │ Plugin  │  │  Tool  │  │  Bindings  │
└───────┘  └─────────┘  └────────┘  └────────────┘
```

### Advantages

✅ **Single source of truth**: Algorithm implemented once
✅ **Easier testing**: Test core library independently
✅ **Consistent results**: All interfaces produce identical output
✅ **Easier maintenance**: Bug fixes apply to all wrappers
✅ **Modular development**: Can add new wrappers without touching core
✅ **Better code quality**: Core can be thoroughly optimized and tested

### Disadvantages

⚠️ **Initial overhead**: More upfront design work
⚠️ **Build complexity**: Need to manage multiple build targets
⚠️ **API design**: Need stable interface between core and wrappers

### Verdict

**Strongly recommended**. This is industry best practice for plugin development.

---

## Python as Core Implementation

### Can Python Be Compiled to a Library?

**Yes!** Several options exist:

---

### Option 1: Cython

**What it is**: Compile Python (with type annotations) to C extension modules

**How it works**:
```python
# vinyl_core.pyx (Cython source)
import numpy as np
cimport numpy as np

def detect_clicks(np.ndarray[np.float32_t, ndim=1] audio,
                  float threshold):
    # Type annotations enable C-speed compilation
    cdef int i
    cdef list clicks = []

    for i in range(len(audio)):
        if audio[i] > threshold:
            clicks.append(i)

    return clicks
```

**Compile to shared library**:
```bash
cython vinyl_core.pyx --cplus
g++ -shared -fPIC vinyl_core.cpp -o libvinylcore.so \
    -I/usr/include/python3.9 -lpython3.9
```

**Advantages**:
- ✅ Can keep Python code mostly as-is
- ✅ Type annotations give C-like performance
- ✅ Gradual optimization (start pure Python, add types where needed)
- ✅ Can still use NumPy/SciPy (they're already compiled)
- ✅ Familiar Python syntax

**Disadvantages**:
- ⚠️ Still requires Python runtime (libpython)
- ⚠️ NumPy/SciPy dependencies must be available
- ⚠️ Larger binary size (includes Python interpreter)
- ⚠️ May complicate distribution

**Feasibility**: ⭐⭐⭐⭐ Good option if Python familiarity is priority

---

### Option 2: PyBind11

**What it is**: Create C++ bindings for Python code, or expose C++ to Python

**Typical use**: Write core in C++, expose to Python

```cpp
// C++ implementation
#include <vector>
#include <pybind11/pybind11.h>

std::vector<int> detect_clicks(const std::vector<float>& audio,
                                float threshold) {
    std::vector<int> clicks;
    for (size_t i = 0; i < audio.size(); i++) {
        if (audio[i] > threshold) {
            clicks.push_back(i);
        }
    }
    return clicks;
}

// Python binding
PYBIND11_MODULE(vinyl_core, m) {
    m.def("detect_clicks", &detect_clicks);
}
```

**Advantages**:
- ✅ C++ performance
- ✅ Easy to create Python bindings
- ✅ Can gradually port from Python to C++
- ✅ Modern C++ syntax (easier than pure C)

**Disadvantages**:
- ⚠️ Need to learn C++ (though similar to Python for basic use)
- ⚠️ More verbose than Python

**Feasibility**: ⭐⭐⭐⭐⭐ Excellent option for gradual migration

---

### Option 3: Embedded Python Interpreter

**What it is**: C/C++ plugin embeds Python interpreter, calls Python code

```cpp
#include <Python.h>

void process_audio(float* audio, size_t length) {
    Py_Initialize();

    // Import Python module
    PyObject* module = PyImport_ImportModule("vinyl_scratch_removal");
    PyObject* func = PyObject_GetAttrString(module, "process");

    // Convert audio to Python array
    PyObject* py_audio = /* conversion */;

    // Call Python function
    PyObject* result = PyObject_CallObject(func, py_audio);

    // Convert result back to C array
    /* ... */

    Py_Finalize();
}
```

**Advantages**:
- ✅ Can use existing Python code without changes
- ✅ Full access to Python ecosystem

**Disadvantages**:
- ⚠️ **Major performance hit**: Interpreter overhead + Python/C conversion
- ⚠️ Requires Python runtime on user's system
- ⚠️ Complex dependency management
- ⚠️ Large binary distribution
- ⚠️ Not suitable for real-time audio

**Feasibility**: ⭐⭐ Technically possible but not recommended for audio plugins

---

### Option 4: Nuitka / PyInstaller

**What it is**: Package Python as standalone executable

**Advantages**:
- ✅ User doesn't need Python installed
- ✅ Can keep Python code

**Disadvantages**:
- ⚠️ Creates executable, not library
- ⚠️ Can't easily create plugin (.so/.dll) for LV2/LADSPA
- ⚠️ Large binary size (bundles entire Python + dependencies)
- ⚠️ Still interpreter overhead

**Feasibility**: ⭐⭐ Good for CLI tool, not for plugins

---

## Architecture Designs

### Architecture A: Pure C++ Core

**Structure**:
```
libvinyl-core/               # C++ core library
├── src/
│   ├── detection.cpp        # Click detection
│   ├── interpolation.cpp    # AR interpolation
│   ├── processor.cpp        # Main processor
│   └── utils.cpp            # Utilities
├── include/
│   └── vinyl_core.h         # Public API
└── CMakeLists.txt

lv2-plugin/                  # LV2 wrapper
├── vinyl_lv2.c
└── CMakeLists.txt

ladspa-plugin/               # LADSPA wrapper
├── vinyl_ladspa.c
└── Makefile

cli-tool/                    # Command-line tool
├── main.cpp
└── CMakeLists.txt

python-bindings/             # PyBind11 bindings
├── bindings.cpp
└── setup.py
```

**Public API** (`vinyl_core.h`):
```cpp
#ifndef VINYL_CORE_H
#define VINYL_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VinylProcessor VinylProcessor;

// Create/destroy processor
VinylProcessor* vinyl_processor_create(float sample_rate);
void vinyl_processor_destroy(VinylProcessor* proc);

// Set parameters
void vinyl_processor_set_threshold(VinylProcessor* proc, float threshold);
void vinyl_processor_set_mode(VinylProcessor* proc, int mode);

// Process audio
int vinyl_processor_process(VinylProcessor* proc,
                            float* audio,
                            size_t length);

#ifdef __cplusplus
}
#endif

#endif
```

**Advantages**:
- ✅ Maximum performance
- ✅ No runtime dependencies (except math library)
- ✅ Small binary size
- ✅ Works on any platform
- ✅ Industry standard

**Disadvantages**:
- ⚠️ Need to port Python algorithm to C++
- ⚠️ C++ learning curve

---

### Architecture B: Cython Core

**Structure**:
```
vinyl-core/                  # Cython core
├── vinyl_core.pyx          # Main algorithm (Cython)
├── detection.pyx           # Detection (Cython)
├── interpolation.pyx       # Interpolation (Cython)
└── setup.py                # Build script

lv2-plugin/                  # LV2 wrapper (C + embedded Python)
├── vinyl_lv2.c
└── Makefile

cli-tool/                    # CLI wrapper (Python)
├── vinyl_cli.py
└── setup.py
```

**Cython example** (can still use NumPy):
```python
# detection.pyx
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def detect_clicks(np.ndarray[np.float32_t, ndim=1] audio,
                  float threshold,
                  int window_size):
    """Detect clicks using second derivative."""

    cdef int i, length = len(audio)
    cdef float diff1, diff2
    cdef list clicks = []

    # Fast C-speed loop
    for i in range(2, length):
        diff1 = audio[i] - audio[i-1]
        diff2 = diff1 - (audio[i-1] - audio[i-2])

        if abs(diff2) > threshold:
            clicks.append(i)

    return clicks
```

**Advantages**:
- ✅ Keep Python code familiarity
- ✅ Good performance with type annotations
- ✅ Can still use NumPy/SciPy
- ✅ Gradual optimization

**Disadvantages**:
- ⚠️ Requires Python runtime for plugins
- ⚠️ Larger distribution (Python + NumPy)
- ⚠️ More complex deployment

---

### Architecture C: Hybrid C++ Core + Python Reference

**Structure**:
```
libvinyl-core/               # C++ core (production)
├── src/
│   └── [C++ implementation]
└── include/

python-reference/            # Python reference (development/testing)
├── vinyl_scratch_removal.py
└── tests/

wrappers/
├── lv2/                     # Links to C++ core
├── ladspa/                  # Links to C++ core
├── cli/                     # Links to C++ core
└── python/                  # PyBind11 bindings to C++ core
```

**Workflow**:
1. Develop algorithm in Python (fast iteration)
2. Test thoroughly
3. Port to C++ when stable
4. Use Python to generate test vectors for C++ validation
5. Keep Python as reference implementation

**Advantages**:
- ✅ Best of both worlds
- ✅ Python for development, C++ for production
- ✅ Python serves as specification/test oracle
- ✅ Maximum performance in production

**Disadvantages**:
- ⚠️ Maintain two implementations
- ⚠️ Initial porting effort

---

## Implementation Roadmap

### Phase 1: Prototype (Python)

**Status**: ✅ **DONE** - We already have this!

Current `vinyl_scratch_removal.py` serves as reference implementation.

---

### Phase 2: Core Library Decision

**Choose architecture based on priorities:**

#### Option 2A: If Python familiarity is critical

**Use Cython approach**:

```bash
# Install Cython
pip install cython

# Convert Python to Cython with type annotations
# vinyl_core.pyx
```

**Benefits**:
- Leverage existing Python code
- Gradual optimization
- Still fast (near C-speed with types)

**Trade-off**:
- Requires Python runtime
- Larger distribution

---

#### Option 2B: If maximum portability/performance needed

**Port to C++ using Eigen**:

```cpp
// Very similar to Python/NumPy syntax!
#include <Eigen/Dense>

Eigen::VectorXd detect_clicks(const Eigen::VectorXf& audio) {
    // Nearly identical to Python logic
    Eigen::VectorXf diff1 = audio.tail(n-1) - audio.head(n-1);
    Eigen::VectorXf diff2 = diff1.tail(n-2) - diff1.head(n-2);
    // ...
}
```

**Benefits**:
- No runtime dependencies
- Maximum performance
- Small binary
- Industry standard

**Trade-off**:
- Need to learn some C++
- Porting effort (~1-2 weeks)

---

### Phase 3: Core Library Implementation

**Regardless of choice, create clean API**:

```c
// vinyl_core.h - C API (callable from any language)

typedef enum {
    VINYL_MODE_CONSERVATIVE = 0,
    VINYL_MODE_STANDARD = 1,
    VINYL_MODE_AGGRESSIVE = 2
} VinylMode;

typedef struct {
    float sample_rate;
    float threshold;
    int ar_order;
    VinylMode mode;
} VinylConfig;

// Create processor
void* vinyl_create(VinylConfig config);

// Process audio (in-place)
int vinyl_process(void* processor, float* audio, size_t frames);

// Destroy processor
void vinyl_destroy(void* processor);
```

**Why C API?**:
- Can be called from C++, Python, LV2, LADSPA, etc.
- Maximum compatibility
- Clear ABI (application binary interface)

---

### Phase 4: Wrappers

Once core library exists, create wrappers:

#### LV2 Plugin Wrapper

```c
// vinyl_lv2.c
#include <lv2/lv2plug.in/ns/lv2core/lv2.h>
#include "vinyl_core.h"

typedef struct {
    void* processor;
    const float* input;
    float* output;
    const float* threshold;
} VinylLV2;

static void run(LV2_Handle instance, uint32_t n_samples) {
    VinylLV2* plugin = (VinylLV2*)instance;

    // Copy input to output
    memcpy(plugin->output, plugin->input, n_samples * sizeof(float));

    // Process in-place
    vinyl_process(plugin->processor, plugin->output, n_samples);
}
```

#### CLI Tool Wrapper

```cpp
// main.cpp
#include "vinyl_core.h"
#include <sndfile.h>

int main(int argc, char** argv) {
    // Parse arguments
    // Load audio with libsndfile

    VinylConfig config = {
        .sample_rate = 44100,
        .threshold = 3.0,
        .ar_order = 20,
        .mode = VINYL_MODE_STANDARD
    };

    void* processor = vinyl_create(config);
    vinyl_process(processor, audio_buffer, num_frames);
    vinyl_destroy(processor);

    // Save output
}
```

#### Python Bindings (PyBind11)

```cpp
// python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vinyl_core.h"

namespace py = pybind11;

py::array_t<float> process_audio(py::array_t<float> audio,
                                   float threshold) {
    auto buf = audio.request();
    float* ptr = (float*) buf.ptr;

    VinylConfig config = {44100, threshold, 20, VINYL_MODE_STANDARD};
    void* proc = vinyl_create(config);
    vinyl_process(proc, ptr, buf.size);
    vinyl_destroy(proc);

    return audio;
}

PYBIND11_MODULE(vinyl_core_native, m) {
    m.def("process_audio", &process_audio);
}
```

---

## Recommended Approach

### For Your Use Case (Python Familiarity)

**Recommended: Start with Cython, consider C++ later**

#### Step 1: Convert Python to Cython (Gradual)

```python
# Start with pure Python (what we have)
def detect_clicks(audio, threshold):
    clicks = []
    for i in range(len(audio)):
        if audio[i] > threshold:
            clicks.append(i)
    return clicks

# Add type annotations (still valid Python!)
def detect_clicks(audio: np.ndarray, threshold: float) -> list:
    clicks: list = []
    i: int
    for i in range(len(audio)):
        if audio[i] > threshold:
            clicks.append(i)
    return clicks

# Convert to Cython (.pyx file, add cimport)
import numpy as np
cimport numpy as np

def detect_clicks(np.ndarray[np.float32_t, ndim=1] audio,
                  float threshold):
    cdef list clicks = []
    cdef int i
    for i in range(len(audio)):
        if audio[i] > threshold:
            clicks.append(i)
    return clicks
```

Each step provides incremental speedup!

#### Step 2: Build as Shared Library

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "vinyl_core",
        ["vinyl_core.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="vinyl-core",
    ext_modules=cythonize(extensions, language_level=3),
)
```

```bash
python setup.py build_ext --inplace
# Creates vinyl_core.so (Linux) or vinyl_core.pyd (Windows)
```

#### Step 3: Create LV2 Wrapper

Use Python embedding or create thin C wrapper:

```c
// Option A: Embed Python (simpler, less portable)
#include <Python.h>

void* vinyl_create() {
    Py_Initialize();
    PyObject* module = PyImport_ImportModule("vinyl_core");
    return (void*)module;
}

// Option B: Port just the wrapper layer to C
// Call Cython-compiled module via C API
```

#### Step 4: Evaluate

After Cython implementation:
- Measure performance
- Test distribution
- If satisfied → Done!
- If need more performance → Port hot paths to C++

---

### Alternative: Pure C++ from Start

**If you're willing to learn some C++**, it's not as scary as it seems:

**Python NumPy**:
```python
import numpy as np
audio = np.array([1.0, 2.0, 3.0])
result = audio * 2.0
```

**C++ Eigen** (very similar!):
```cpp
#include <Eigen/Dense>
Eigen::VectorXf audio(3);
audio << 1.0, 2.0, 3.0;
Eigen::VectorXf result = audio * 2.0;
```

**Python loop**:
```python
for i in range(len(audio)):
    if audio[i] > threshold:
        clicks.append(i)
```

**C++ loop**:
```cpp
for (int i = 0; i < audio.size(); i++) {
    if (audio[i] > threshold) {
        clicks.push_back(i);
    }
}
```

**Very similar syntax!**

---

## Practical Migration Path

### Week 1-2: Cython Conversion

1. Copy `vinyl_scratch_removal.py` to `vinyl_core.pyx`
2. Add type annotations gradually
3. Test performance improvements
4. Build shared library

### Week 3: Simple CLI Wrapper

Create minimal C wrapper that calls Cython library:

```c
#include "vinyl_core.h"  // Your C API
#include <dlfcn.h>        // Dynamic loading

int main() {
    void* lib = dlopen("vinyl_core.so", RTLD_LAZY);
    // Call Cython functions via function pointers
}
```

### Week 4: LV2 Plugin

Use LV2 templates, link to your core library.

### Week 5+: Optimization

- Profile hot paths
- Port critical sections to C++ if needed
- Keep Python as reference

---

## Build System

### CMake for C++ Core

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(VinylCore)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Core library
add_library(vinyl_core SHARED
    src/detection.cpp
    src/interpolation.cpp
    src/processor.cpp
)

target_include_directories(vinyl_core PUBLIC
    include/
    ${EIGEN3_INCLUDE_DIR}
)

# LV2 plugin
add_library(vinyl_lv2 MODULE
    lv2/vinyl_lv2.c
)
target_link_libraries(vinyl_lv2 vinyl_core)

# CLI tool
add_executable(vinyl_cli
    cli/main.cpp
)
target_link_libraries(vinyl_cli vinyl_core sndfile)
```

### Setup.py for Cython

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="vinyl-core",
    ext_modules=cythonize("vinyl_core.pyx"),
    install_requires=['numpy', 'scipy'],
)
```

---

## Summary

### Direct Answer to Your Question

**Q: Should I write a core library with separate wrappers?**
**A: Yes! ✅** Excellent architectural decision.

**Q: Can Python be compiled to a library instead of C++?**
**A: Yes! ✅** Via Cython - you can keep Python syntax.

### Recommended Path

**For someone more familiar with Python:**

1. **Phase 1**: Convert Python to Cython with type annotations
2. **Phase 2**: Build as shared library (.so/.dll)
3. **Phase 3**: Create C wrapper API around Cython library
4. **Phase 4**: Build LV2/LADSPA wrappers calling C API
5. **Phase 5**: (Optional) Port performance-critical parts to C++ if needed

**Timeline**: 2-4 weeks to working LV2 plugin using Cython core

**Learning curve**: Gentle - mostly Python with gradual C type additions

### Ultimate Architecture

```
┌────────────────────────────────────────┐
│  Core Library (Cython or C++)          │
│  - Click detection                      │
│  - AR interpolation                     │
│  - C API for interoperability          │
└────────────────────────────────────────┘
              │
    ┌─────────┼─────────┬──────────┐
    │         │         │          │
┌───▼───┐ ┌──▼──┐ ┌────▼─────┐ ┌──▼────┐
│ LV2   │ │ CLI │ │ LADSPA   │ │Python │
│Plugin │ │Tool │ │ Plugin   │ │Wrapper│
└───────┘ └─────┘ └──────────┘ └───────┘
```

**Result**: Write algorithm once, use everywhere!

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Author**: Claude (Anthropic)
