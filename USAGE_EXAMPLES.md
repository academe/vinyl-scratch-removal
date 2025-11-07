# Usage Examples

This guide provides practical examples for using the Vinyl Scratch Removal tools.

## Nyquist Plugin Examples

### Example 1: Light Cleaning of Modern Vinyl

**Scenario**: Recently pressed vinyl with occasional light clicks.

**Settings**:
- Click Sensitivity: 15
- Maximum Click Width: 2.0 ms
- Detection Mode: Clicks Only

**Steps**:
1. Open your audio in Audacity
2. Select all (Ctrl+A / Cmd+A)
3. Effect > Vinyl Scratch Removal
4. Set parameters as above
5. Preview to verify
6. Apply

**Result**: Removes obvious clicks while preserving all musical content.

---

### Example 2: Standard Restoration of Older Vinyl

**Scenario**: 1970s-80s vinyl with regular clicks and some pops.

**Settings**:
- Click Sensitivity: 25
- Maximum Click Width: 3.0 ms
- Detection Mode: Clicks + Crackle

**Steps**:
1. Open audio in Audacity
2. Select the portion to process (or all)
3. Effect > Vinyl Scratch Removal
4. Set parameters as above
5. Preview a section with visible clicks
6. Adjust if needed (listen for over-processing)
7. Apply

**Result**: Removes most clicks and pops without damaging audio.

---

### Example 3: Heavy Restoration of Damaged Vinyl

**Scenario**: Old, scratched vinyl from the 1950s-60s with heavy surface noise.

**Settings**:
- Click Sensitivity: 40
- Maximum Click Width: 5.0 ms
- Detection Mode: Aggressive

**Steps**:
1. Open audio in Audacity
2. First pass:
   - Select all
   - Effect > Vinyl Scratch Removal
   - Sensitivity: 30, Width: 3.0 ms, Mode: Standard
   - Apply
3. Second pass (for remaining clicks):
   - Effect > Vinyl Scratch Removal
   - Sensitivity: 40, Width: 5.0 ms, Mode: Aggressive
   - Apply
4. Optional: Apply gentle noise reduction for hiss

**Result**: Maximum click removal, suitable for heavily damaged records.

**Warning**: Aggressive settings may introduce artifacts. Always compare with original.

---

### Example 4: Processing Classical Music

**Scenario**: Classical vinyl with dynamic range (quiet to loud passages).

**Settings**:
- Click Sensitivity: 18
- Maximum Click Width: 2.0 ms
- Detection Mode: Clicks + Crackle

**Special considerations**:
- Process in sections (quiet and loud separately)
- For quiet passages: Lower sensitivity (12-15)
- For loud passages: Higher sensitivity (20-25)

**Steps**:
1. Find quiet passage, select it
2. Effect > Vinyl Scratch Removal
3. Sensitivity: 12-15, Width: 2.0 ms
4. Apply
5. Find loud passage, select it
6. Effect > Vinyl Scratch Removal
7. Sensitivity: 20-25, Width: 2.0 ms
8. Apply
9. Process middle-level sections with standard settings

**Result**: Consistent click removal without damaging delicate passages.

---

## Python Tool Examples

### Example 1: Basic Processing

**Command**:
```bash
python vinyl_scratch_removal.py original.wav cleaned.wav
```

**What it does**:
- Uses default settings (standard mode, threshold 3.0)
- Suitable for most recordings

---

### Example 2: Conservative Processing

**Scenario**: High-quality recording with just a few clicks.

**Command**:
```bash
python vinyl_scratch_removal.py original.wav cleaned.wav \
  --mode conservative \
  --threshold 4.0 \
  --max-width 1.5
```

**Parameters explained**:
- `--mode conservative`: Only detect obvious clicks
- `--threshold 4.0`: Higher threshold = less sensitive
- `--max-width 1.5`: Only detect very short clicks

---

### Example 3: Aggressive Processing

**Scenario**: Heavily damaged record with many clicks.

**Command**:
```bash
python vinyl_scratch_removal.py original.wav cleaned.wav \
  --mode aggressive \
  --threshold 2.0 \
  --max-width 5.0
```

**Parameters explained**:
- `--mode aggressive`: Maximum detection
- `--threshold 2.0`: Lower threshold = more sensitive
- `--max-width 5.0`: Detect longer clicks and pops

---

### Example 4: Fine-Tuning AR Interpolation

**Scenario**: You want the highest quality interpolation.

**Command**:
```bash
python vinyl_scratch_removal.py original.wav cleaned.wav \
  --threshold 3.0 \
  --ar-order 30
```

**Parameters explained**:
- `--ar-order 30`: Higher order AR model (default is 20)
- Better interpolation quality, but slower processing

**Note**: Higher AR orders (30-50) provide better quality for tonal music but take longer to process.

---

### Example 5: Batch Processing

**Scenario**: Process multiple files with the same settings.

**Bash script** (Linux/macOS):
```bash
#!/bin/bash
for file in *.wav; do
  echo "Processing $file..."
  python vinyl_scratch_removal.py "$file" "cleaned_$file" \
    --mode standard \
    --threshold 3.0
done
```

**Windows batch** (save as process_all.bat):
```batch
@echo off
for %%f in (*.wav) do (
  echo Processing %%f...
  python vinyl_scratch_removal.py "%%f" "cleaned_%%f" --mode standard --threshold 3.0
)
```

---

### Example 6: Processing Specific Sections

**Scenario**: Only one section has clicks, rest is clean.

**Steps**:
1. Use Audacity to extract the problematic section
2. Export as WAV (e.g., `section.wav`)
3. Process it:
   ```bash
   python vinyl_scratch_removal.py section.wav section_cleaned.wav
   ```
4. Import back into Audacity
5. Replace the original section

---

## Comparison Examples

### Before/After Analysis

**Recommended workflow**:

1. **Make a copy**:
   ```bash
   cp original.wav original_backup.wav
   ```

2. **Process with standard settings**:
   ```bash
   python vinyl_scratch_removal.py original.wav cleaned_standard.wav
   ```

3. **Process with aggressive settings**:
   ```bash
   python vinyl_scratch_removal.py original.wav cleaned_aggressive.wav \
     --mode aggressive --threshold 2.5
   ```

4. **Compare in Audacity**:
   - Import all three files
   - Use "Transport > Play > One Second" to quickly compare
   - Zoom in on waveform to see clicks removed
   - Listen for any artifacts

5. **Choose the best result**

---

## Troubleshooting Examples

### Problem: Still hearing clicks after processing

**Try this progression**:

```bash
# First attempt - standard
python vinyl_scratch_removal.py input.wav output1.wav

# If clicks remain - more sensitive
python vinyl_scratch_removal.py input.wav output2.wav --threshold 2.5

# If still there - aggressive
python vinyl_scratch_removal.py input.wav output3.wav \
  --mode aggressive --threshold 2.0

# If STILL there - maximum
python vinyl_scratch_removal.py input.wav output4.wav \
  --mode aggressive --threshold 1.5 --max-width 8.0
```

---

### Problem: Audio sounds "watery" or damaged

**Try reducing sensitivity**:

```bash
# Too aggressive, back off
python vinyl_scratch_removal.py input.wav output.wav \
  --mode conservative --threshold 4.5 --max-width 1.0

# Or just process obvious clicks
python vinyl_scratch_removal.py input.wav output.wav \
  --threshold 5.0
```

---

### Problem: Only want to process specific frequencies

**Use Audacity preprocessing**:

1. Open audio in Audacity
2. Apply high-pass filter (Effect > High-Pass Filter, 20 Hz)
   - Removes rumble before click removal
3. Export as WAV
4. Process with Python tool:
   ```bash
   python vinyl_scratch_removal.py filtered.wav final.wav
   ```

---

## Genre-Specific Recommendations

### Jazz

- **Settings**: Conservative to standard
- **Why**: Preserve brush strokes and cymbals
- **Command**:
  ```bash
  python vinyl_scratch_removal.py jazz.wav jazz_clean.wav \
    --mode conservative --threshold 3.5
  ```

### Classical

- **Settings**: Variable by section (quiet vs loud)
- **Why**: Preserve dynamic range
- **Approach**: Process sections separately in Audacity with different sensitivity

### Rock/Pop

- **Settings**: Standard to aggressive
- **Why**: More forgiving of processing artifacts
- **Command**:
  ```bash
  python vinyl_scratch_removal.py rock.wav rock_clean.wav \
    --mode standard --threshold 3.0
  ```

### Spoken Word / Audiobooks

- **Settings**: Aggressive (voice is very tolerant)
- **Why**: Clicks are very distracting in speech
- **Command**:
  ```bash
  python vinyl_scratch_removal.py speech.wav speech_clean.wav \
    --mode aggressive --threshold 2.0
  ```

---

## Advanced Workflows

### Two-Pass Processing

**For best results with heavily damaged records**:

**First pass** - Remove obvious clicks:
```bash
python vinyl_scratch_removal.py original.wav pass1.wav \
  --mode standard --threshold 3.0
```

**Second pass** - Remove remaining clicks:
```bash
python vinyl_scratch_removal.py pass1.wav final.wav \
  --mode aggressive --threshold 3.5
```

**Why**: First pass gets the easy ones, second pass is more targeted.

---

### Combined with Other Processing

**Complete restoration workflow**:

1. **Remove DC offset** (Audacity: Effect > Normalize, check "Remove DC offset")
2. **Export as WAV**
3. **Remove clicks** (Python tool):
   ```bash
   python vinyl_scratch_removal.py step1.wav step2.wav
   ```
4. **Import back to Audacity**
5. **Noise reduction** (Effect > Noise Reduction)
6. **Normalize** (Effect > Normalize)
7. **Export final**

---

## Quick Reference

| Scenario | Nyquist Settings | Python Command |
|----------|------------------|----------------|
| Modern vinyl, few clicks | Sens: 10-15, Width: 2.0ms, Mode: Clicks Only | `--mode conservative --threshold 4.0` |
| Standard restoration | Sens: 20-25, Width: 2.0-3.0ms, Mode: Clicks + Crackle | `--mode standard --threshold 3.0` |
| Heavy damage | Sens: 30-40, Width: 3.0-5.0ms, Mode: Aggressive | `--mode aggressive --threshold 2.0` |
| Delicate/classical | Sens: 15-20, Width: 2.0ms, Mode: Clicks + Crackle | `--mode conservative --threshold 3.5` |
| Spoken word | Sens: 35-45, Width: 2.0-4.0ms, Mode: Aggressive | `--mode aggressive --threshold 2.0` |

---

## Tips for Best Results

1. **Always keep originals** - Never overwrite your source files
2. **Start conservative** - You can always process more aggressively
3. **Preview first** - Test settings on a small section
4. **Listen carefully** - Use good headphones to detect artifacts
5. **Compare A/B** - Switch between original and processed to verify improvement
6. **Process in sections** - Different parts may need different settings
7. **Document your settings** - Keep notes on what worked for future reference

---

For more information, see README.md and INSTALL.md
