# Build Instructions

Complete instructions for building the vinyl scratch removal library, LV2 plugin, and CLI tool.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Building the Core Library](#building-the-core-library)
4. [Building the LV2 Plugin](#building-the-lv2-plugin)
5. [Building the CLI Tool](#building-the-cli-tool)
6. [Testing](#testing)
7. [Installation](#installation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python 3.7+** with development headers
- **GCC** or compatible C compiler
- **Cython** (`pip install cython`)
- **NumPy** (`pip install numpy`)
- **SciPy** (`pip install scipy`)
- **soundfile** (`pip install soundfile`)

### Optional (for LV2 plugin)

- **LV2 development headers** (`lv2-dev` package on Debian/Ubuntu)
- **make**

### Operating System Specific

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3-dev python3-pip gcc make lv2-dev libsndfile1-dev
pip3 install cython numpy scipy soundfile
```

#### Fedora/RHEL

```bash
sudo dnf install python3-devel python3-pip gcc make lv2-devel libsndfile-devel
pip3 install cython numpy scipy soundfile
```

#### Arch Linux

```bash
sudo pacman -S python python-pip gcc make lv2 libsndfile
pip install cython numpy scipy soundfile
```

#### macOS

```bash
brew install python gcc make lv2 libsndfile
pip3 install cython numpy scipy soundfile
```

---

## Quick Start

**Build everything:**

```bash
# 1. Build core library
cd core
python3 setup.py build_ext --inplace
cd ..

# 2. Build LV2 plugin (optional)
cd lv2
make
cd ..

# 3. Test CLI tool
chmod +x cli/vinyl
./cli/vinyl --help
```

---

## Building the Core Library

The core library is written in Cython and compiles to native Python extensions.

### Step 1: Navigate to core directory

```bash
cd core
```

### Step 2: Build

```bash
python3 setup.py build_ext --inplace
```

**What this does:**
- Compiles `detection.pyx` → `detection.so` (or `.pyd` on Windows)
- Compiles `interpolation.pyx` → `interpolation.so`
- Compiles `vinyl_core.pyx` → `vinyl_core.so`

### Step 3: Verify

```bash
python3 -c "import vinyl_core; print('Success!')"
```

If you see "Success!", the build worked!

### Build Options

**Debug build** (with symbols):
```bash
python3 setup.py build_ext --inplace --debug
```

**Optimized build** (default):
```bash
python3 setup.py build_ext --inplace
```

**Install system-wide** (optional):
```bash
python3 setup.py install
```

### Troubleshooting Core Build

**Error: "Python.h not found"**
```bash
# Ubuntu/Debian
sudo apt install python3-dev

# Fedora
sudo dnf install python3-devel
```

**Error: "numpy/arrayobject.h not found"**
```bash
pip3 install numpy
```

**Error: "Cython not found"**
```bash
pip3 install cython
```

---

## Building the LV2 Plugin

The LV2 plugin is a compiled shared library that Audacity and other DAWs can load.

### Step 1: Build core library first

```bash
cd core
python3 setup.py build_ext --inplace
cd ..
```

### Step 2: Navigate to LV2 directory

```bash
cd lv2
```

### Step 3: Build plugin

```bash
make
```

**What this does:**
- Compiles `vinyl_scratch_removal.c` to `vinyl_scratch_removal.so`
- Creates `vinyl_scratch_removal.lv2/` bundle with all required files
- Copies manifest and TTL files

### Step 4: Verify

```bash
ls vinyl_scratch_removal.lv2/
```

You should see:
- `vinyl_scratch_removal.so` (compiled plugin)
- `manifest.ttl` (LV2 manifest)
- `vinyl_scratch_removal.ttl` (plugin description)

### Installation

**Install for current user:**
```bash
make install PREFIX=$HOME/.local
```

**Install system-wide:**
```bash
sudo make install
```

**Manual installation:**
```bash
# Linux
mkdir -p ~/.lv2
cp -r vinyl_scratch_removal.lv2 ~/.lv2/

# macOS
mkdir -p ~/Library/Audio/Plug-Ins/LV2
cp -r vinyl_scratch_removal.lv2 ~/Library/Audio/Plug-Ins/LV2/
```

### Testing in Audacity

1. Restart Audacity
2. Go to `Effect > Add/Remove Plugins`
3. Enable "Vinyl Scratch Removal" if it appears
4. Use from `Effect > Vinyl Scratch Removal`

### Troubleshooting LV2 Build

**Error: "lv2/lv2plug.in/ns/lv2core/lv2.h: No such file"**
```bash
# Ubuntu/Debian
sudo apt install lv2-dev

# Fedora
sudo dnf install lv2-devel

# macOS
brew install lv2
```

**Plugin doesn't appear in Audacity:**
- Check installation path: `ls ~/.lv2/` or `ls /usr/local/lib/lv2/`
- Check Audacity's LV2 scan: `Tools > Plugin Manager`
- Try rescanning: `Tools > Plugin Manager > Rescan`

---

## Building the CLI Tool

The CLI tool is a Python script that uses the core library.

### Step 1: Build core library first

```bash
cd core
python3 setup.py build_ext --inplace
cd ..
```

### Step 2: Make CLI executable

```bash
chmod +x cli/vinyl
```

### Step 3: Test

```bash
./cli/vinyl --help
```

You should see the help message!

### Installation (Optional)

**Install for current user:**
```bash
cp cli/vinyl ~/.local/bin/
cp cli/vinyl_cli.py ~/.local/bin/
```

**Install system-wide:**
```bash
sudo cp cli/vinyl /usr/local/bin/
sudo cp cli/vinyl_cli.py /usr/local/bin/
```

### Usage

```bash
# Basic usage
./cli/vinyl input.wav output.wav

# With options
./cli/vinyl input.wav output.wav --threshold 2.5 --mode aggressive

# Show help
./cli/vinyl --help
```

---

## Testing

### Test Core Library

```bash
cd core
python3 << EOF
import numpy as np
from vinyl_core import VinylProcessor

# Create test audio with a click
audio = np.random.randn(44100).astype(np.float32) * 0.1
audio[1000:1005] = 1.0  # Add click

# Process
processor = VinylProcessor(44100.0, 3.0)
processed = processor.process(audio.copy())

print(f"Original peak at click: {audio[1000:1005].max()}")
print(f"Processed peak at click: {processed[1000:1005].max()}")
print("Test passed!" if processed[1000:1005].max() < 0.5 else "Test failed!")
EOF
```

### Test CLI Tool

```bash
# Generate test file
python3 << EOF
import numpy as np
import soundfile as sf

# Create test audio
audio = np.random.randn(44100).astype(np.float32) * 0.1
audio[1000:1005] = 1.0  # Add click

sf.write('test_input.wav', audio, 44100)
print("Created test_input.wav")
EOF

# Process
./cli/vinyl test_input.wav test_output.wav

# Check output
python3 << EOF
import soundfile as sf
audio, sr = sf.read('test_output.wav')
print(f"Output peak: {audio.max()}")
print("Test passed!" if audio.max() < 0.5 else "Test failed!")
EOF

# Cleanup
rm test_input.wav test_output.wav
```

### Test LV2 Plugin

```bash
# Check if plugin is recognized
lv2ls | grep vinyl

# OR use lv2info
lv2info http://github.com/anthropics/vinyl-scratch-removal
```

---

## Installation

### Complete Installation

**1. Build everything:**
```bash
# Core library
cd core && python3 setup.py build_ext --inplace && cd ..

# LV2 plugin
cd lv2 && make && cd ..

# CLI tool
chmod +x cli/vinyl
```

**2. Install:**
```bash
# Core library (optional, for system-wide Python import)
cd core && python3 setup.py install && cd ..

# LV2 plugin (for Audacity)
cd lv2 && make install PREFIX=$HOME/.local && cd ..

# CLI tool (for command-line use)
mkdir -p ~/.local/bin
cp cli/vinyl ~/.local/bin/
cp cli/vinyl_cli.py ~/.local/bin/
```

**3. Update PATH (if needed):**
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Uninstallation

**Core library:**
```bash
pip3 uninstall vinyl-core
```

**LV2 plugin:**
```bash
cd lv2
make uninstall PREFIX=$HOME/.local
```

**CLI tool:**
```bash
rm ~/.local/bin/vinyl
rm ~/.local/bin/vinyl_cli.py
```

---

## Troubleshooting

### General Issues

**"Module not found" errors:**
```bash
# Check Python can find modules
cd core
python3 -c "import sys; print(sys.path)"

# Ensure you're in the right directory
pwd  # Should show .../vinyl-scratch-removal/
```

**"Permission denied" on CLI tool:**
```bash
chmod +x cli/vinyl
```

### Performance Issues

**Slow processing:**
- Check you built with optimization: `python3 setup.py build_ext --inplace` (no `--debug`)
- Increase AR order cautiously: `--ar-order 30` (higher = slower but better quality)
- Use conservative mode for faster processing: `--mode conservative`

**High memory usage:**
- Process smaller chunks
- Reduce AR order: `--ar-order 10`

### Build Warnings

**"Warning: Using deprecated NumPy API"**
- Safe to ignore, works fine

**"Compiler optimization flags"**
- These are normal, `-O3` optimization is good

### Platform-Specific

**macOS: "Library not loaded"**
```bash
# Check library paths
otool -L core/vinyl_core.*.so

# May need to set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

**Windows:**
- Use Visual Studio compiler or MinGW
- May need to install Microsoft Visual C++ Build Tools
- Replace `.so` with `.pyd` in commands

---

## Development Workflow

### Rebuilding After Changes

**Modified Cython code (.pyx files):**
```bash
cd core
python3 setup.py build_ext --inplace --force
```

**Modified C code (LV2 plugin):**
```bash
cd lv2
make clean
make
```

**Modified Python code (CLI):**
- No rebuild needed, just run!

### Debugging

**Enable Cython debugging:**
```bash
cd core
python3 setup.py build_ext --inplace --debug
gdb python3
> run -c "import vinyl_core"
```

**Check generated C code:**
```bash
cd core
cython detection.pyx -a
# Creates detection.html showing Python/C interaction
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `cd core && python3 setup.py build_ext --inplace` | Build core library |
| `cd lv2 && make` | Build LV2 plugin |
| `./cli/vinyl input.wav output.wav` | Process audio file |
| `make install PREFIX=$HOME/.local` | Install LV2 plugin |
| `python3 -c "import vinyl_core"` | Test core import |
| `lv2ls \| grep vinyl` | Check LV2 plugin |

---

## Getting Help

If you encounter issues:

1. Check this document's troubleshooting section
2. Ensure all prerequisites are installed
3. Try a clean rebuild: `make clean && make`
4. Check system logs for errors
5. Open an issue on GitHub with:
   - Operating system and version
   - Python version (`python3 --version`)
   - Complete error message
   - Build commands you ran

---

**Last Updated:** 2025-11-08
**Version:** 1.0.0
