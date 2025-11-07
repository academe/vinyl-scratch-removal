# Installation Guide

## Nyquist Plugin for Audacity

### Windows

1. **Locate Audacity's Plug-Ins folder**:
   - Default location: `C:\Program Files\Audacity\Plug-Ins\`
   - Or: `C:\Users\[YourName]\AppData\Roaming\Audacity\Plug-Ins\`

2. **Copy the plugin**:
   - Copy `vinyl-scratch-removal.ny` to the Plug-Ins folder
   - You may need administrator privileges

3. **Enable in Audacity**:
   - Open Audacity
   - Go to `Tools > Plugin Manager`
   - Find "Vinyl Scratch Removal" in the list
   - Select it and click `Enable`
   - Click `OK` to close

4. **Verify installation**:
   - Open an audio file or generate a tone
   - Select some audio
   - Go to `Effect` menu
   - You should see "Vinyl Scratch Removal" in the list

### macOS

1. **Locate Audacity's Plug-Ins folder**:
   ```bash
   ~/Library/Application Support/audacity/Plug-Ins/
   ```

2. **Copy the plugin**:
   ```bash
   cp vinyl-scratch-removal.ny ~/Library/Application\ Support/audacity/Plug-Ins/
   ```

   Or manually:
   - Open Finder
   - Press `Cmd+Shift+G` (Go to Folder)
   - Enter: `~/Library/Application Support/audacity/Plug-Ins/`
   - Copy `vinyl-scratch-removal.ny` there

3. **Enable in Audacity**:
   - Open Audacity
   - Go to `Audacity > Plugin Manager` (or `Tools > Plugin Manager`)
   - Find "Vinyl Scratch Removal" in the list
   - Select it and click `Enable`
   - Click `OK` to close

4. **Verify installation**:
   - Select some audio
   - Go to `Effect > Vinyl Scratch Removal`

### Linux

1. **Locate Audacity's Plug-Ins folder**:
   - User folder: `~/.audacity-data/Plug-Ins/`
   - System folder: `/usr/share/audacity/plug-ins/`
   - Or: `/usr/local/share/audacity/plug-ins/`

2. **Copy the plugin** (user installation):
   ```bash
   # Create folder if it doesn't exist
   mkdir -p ~/.audacity-data/Plug-Ins/

   # Copy plugin
   cp vinyl-scratch-removal.ny ~/.audacity-data/Plug-Ins/
   ```

   Or system-wide (requires sudo):
   ```bash
   sudo cp vinyl-scratch-removal.ny /usr/share/audacity/plug-ins/
   ```

3. **Enable in Audacity**:
   - Open Audacity
   - Go to `Tools > Plugin Manager`
   - Find "Vinyl Scratch Removal" in the list
   - Select it and click `Enable`
   - Click `OK` to close

4. **Verify installation**:
   - Select some audio
   - Go to `Effect > Vinyl Scratch Removal`

### Troubleshooting Nyquist Plugin

**Plugin doesn't appear**:
- Restart Audacity after copying the file
- Check Plugin Manager: `Tools > Plugin Manager`
- Look for "Vinyl Scratch Removal" and enable it
- Verify the file has `.ny` extension (not `.ny.txt`)

**Plugin appears but won't run**:
- Check that the `.ny` file is not corrupted
- Re-download/copy the file
- Check Audacity version (requires 2.3.0+)

**Error messages**:
- Check Audacity's error log: `Help > Show Log`
- Verify the plugin syntax is correct
- Try disabling and re-enabling in Plugin Manager

## Python Tool

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

#### Windows

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Run installer
   - **Important**: Check "Add Python to PATH" during installation

2. **Open Command Prompt**:
   - Press `Win+R`, type `cmd`, press Enter

3. **Navigate to the project folder**:
   ```cmd
   cd C:\path\to\vinyl-scratch-removal
   ```

4. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```cmd
   python vinyl_scratch_removal.py --help
   ```

#### macOS

1. **Install Python** (if not already installed):
   - macOS 10.15+ includes Python 3
   - Or install via Homebrew:
     ```bash
     brew install python3
     ```

2. **Open Terminal**:
   - `Applications > Utilities > Terminal`

3. **Navigate to the project folder**:
   ```bash
   cd ~/path/to/vinyl-scratch-removal
   ```

4. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

   Or:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python3 vinyl_scratch_removal.py --help
   ```

#### Linux

1. **Install Python** (if not already installed):
   ```bash
   # Debian/Ubuntu
   sudo apt update
   sudo apt install python3 python3-pip

   # Fedora
   sudo dnf install python3 python3-pip

   # Arch
   sudo pacman -S python python-pip
   ```

2. **Navigate to the project folder**:
   ```bash
   cd ~/path/to/vinyl-scratch-removal
   ```

3. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

   Or in a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python3 vinyl_scratch_removal.py --help
   ```

### Dependencies

The Python tool requires:

- **numpy** - Numerical computing
- **scipy** - Scientific computing and signal processing
- **soundfile** - Audio file I/O (requires libsndfile)

#### Installing libsndfile

**macOS**:
```bash
brew install libsndfile
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt install libsndfile1
```

**Linux (Fedora)**:
```bash
sudo dnf install libsndfile
```

**Windows**:
- Usually installed automatically with soundfile
- If issues occur, download from [libsndfile.github.io](http://www.mega-nerd.com/libsndfile/)

### Using a Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate
deactivate
```

### Troubleshooting Python Tool

**"Command not found: python"**:
- Try `python3` instead of `python`
- Ensure Python is in PATH
- Reinstall Python with "Add to PATH" option

**"ModuleNotFoundError: No module named 'numpy'"**:
- Install dependencies: `pip install -r requirements.txt`
- Check you're using the correct Python: `which python` or `where python`
- If using virtual environment, ensure it's activated

**"soundfile failed to load"**:
- Install libsndfile (see Dependencies section)
- Try: `pip install --upgrade soundfile`

**"Permission denied"**:
- Use `--user` flag: `pip install --user -r requirements.txt`
- Or use virtual environment
- On Linux: don't use `sudo pip` (use virtual environment instead)

## Quick Start

### Nyquist Plugin

1. Open Audacity
2. Load your vinyl recording
3. Select a portion (or all) of the audio
4. Go to `Effect > Vinyl Scratch Removal`
5. Start with default settings
6. Click `Preview` to hear the result
7. Adjust parameters if needed
8. Click `Apply`

### Python Tool

```bash
# Basic usage
python vinyl_scratch_removal.py input.wav output.wav

# With preview (process a small section first)
python vinyl_scratch_removal.py input.wav test.wav --threshold 3.0
# Listen to test.wav, adjust parameters if needed
```

## Updating

### Nyquist Plugin

1. Delete old `vinyl-scratch-removal.ny` from Plug-Ins folder
2. Copy new version
3. Restart Audacity
4. Plugin Manager will automatically detect the update

### Python Tool

1. Replace `vinyl_scratch_removal.py` with new version
2. Update dependencies if needed:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Uninstallation

### Nyquist Plugin

1. Delete `vinyl-scratch-removal.ny` from Audacity's Plug-Ins folder
2. Restart Audacity

### Python Tool

1. Delete the project folder
2. If using virtual environment, delete `venv` folder
3. If installed system-wide, uninstall packages:
   ```bash
   pip uninstall numpy scipy soundfile
   ```

## Support

If you encounter installation issues:

1. Check this guide's Troubleshooting sections
2. Verify you meet the prerequisites
3. Check for error messages and search for solutions
4. Open an issue on GitHub with:
   - Your operating system and version
   - Python version (`python --version`)
   - Audacity version
   - Complete error message
   - Steps you've already tried

## Version Information

- **Plugin Version**: 1.0.0
- **Minimum Audacity**: 2.3.0 (recommended: 3.0+)
- **Minimum Python**: 3.7
- **Tested on**:
  - Audacity 3.x (Windows, macOS, Linux)
  - Python 3.8, 3.9, 3.10, 3.11
