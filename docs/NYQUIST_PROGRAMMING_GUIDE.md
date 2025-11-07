# Nyquist Programming Guide for Audio Effects

Comprehensive guide to programming Audacity plugins using the Nyquist language, with focus on audio restoration and click removal.

## Table of Contents

1. [Introduction to Nyquist](#introduction-to-nyquist)
2. [Plugin Structure](#plugin-structure)
3. [Audio Processing Concepts](#audio-processing-concepts)
4. [Available Functions](#available-functions)
5. [Limitations and Workarounds](#limitations-and-workarounds)
6. [Best Practices](#best-practices)
7. [Example Implementations](#example-implementations)
8. [Debugging Tips](#debugging-tips)

---

## Introduction to Nyquist

### What is Nyquist?

**Nyquist** is a programming language for sound synthesis and analysis designed by Roger Dannenberg at Carnegie Mellon University.

**Key characteristics**:
- Based on **LISP** (specifically XLISP)
- Named after Harry Nyquist (of Nyquist-Shannon sampling theorem fame)
- Functional programming paradigm
- Designed for expressing audio transformations mathematically
- Integrated into Audacity as plugin language

**Philosophy**:
Nyquist treats audio as continuous signals (mathematical functions) rather than discrete sample arrays. This is fundamentally different from imperative languages like C/Python.

### Resources

**Official Documentation**:
- Nyquist Manual: https://www.cs.cmu.edu/~rbd/doc/nyquist/
- Audacity Nyquist Manual: https://manual.audacityteam.org/man/nyquist.html
- Programming in Nyquist: https://manual.audacityteam.org/man/programming_in_nyquist.html

**Learning Resources**:
- Introduction to Nyquist and Lisp: https://manual.audacityteam.org/man/introduction_to_nyquist_and_lisp_programming.html
- Creating Nyquist Plugins: https://manual.audacityteam.org/man/creating_nyquist_plug_ins.html
- Nyquist Prompt (testing environment in Audacity): Tools > Nyquist Prompt

---

## Plugin Structure

### Minimum Plugin Template

```lisp
;nyquist plug-in
;version 4
;type process
;name "My Effect"

;; Your code here
(mult *track* 0.5)  ; Simple example: reduce volume by 50%
```

### Complete Plugin Template

```lisp
;nyquist plug-in
;version 4
;type process
;preview enabled
;name "Effect Name"
;manpage "Effect_Name"
;debugflags trace
;author "Your Name"
;release 1.0.0
;copyright "License"

;; Control parameters
;control param1 "Parameter 1" int "20" 1 100
;control param2 "Parameter 2" real "2.0" 0.1 10.0
;control param3 "Mode" choice "Mode A,Mode B,Mode C" 0

;; Plugin code
(your-processing-function *track*)
```

### Header Fields

#### Required Headers

```lisp
;nyquist plug-in     ; REQUIRED: Identifies this as a plugin
;version 4           ; REQUIRED: Nyquist version (use 4 for modern Audacity)
;type process        ; REQUIRED: Plugin type
;name "Effect Name"  ; REQUIRED: Name shown in Audacity menu
```

#### Optional Headers

```lisp
;preview enabled     ; Enables preview button
;manpage "Page_Name" ; Links to manual page
;debugflags trace    ; Enables debugging output
;author "Name"       ; Plugin author
;release 1.0.0       ; Version number
;copyright "License" ; License information
```

#### Plugin Types

```lisp
;type process    ; Processes selected audio (most common)
;type generate   ; Generates new audio
;type analyze    ; Analyzes audio, returns text/labels
;type tool       ; Utility function
```

### Control Parameters

Controls create GUI elements for user input.

**Integer control**:
```lisp
;control var-name "Display Name" int "default" min max
;control threshold "Sensitivity" int "20" 1 100
```

**Real (float) control**:
```lisp
;control var-name "Display Name" real "default" min max
;control width "Click Width (ms)" real "2.0" 0.1 10.0
```

**Choice (dropdown) control**:
```lisp
;control var-name "Display Name" choice "Option1,Option2,Option3" default-index
;control mode "Detection Mode" choice "Low,Medium,High" 1
```

**String control**:
```lisp
;control var-name "Display Name" string "default"
;control label-text "Label Text" string "Click detected"
```

**File control**:
```lisp
;control filename "File Name" file "default.txt"
```

---

## Audio Processing Concepts

### Signal Representation

In Nyquist, audio is represented as **SOUND** objects, not arrays.

```lisp
;; Built-in variables
*track*      ; Mono audio (SOUND object)
s            ; Same as *track*
*track*      ; Current track being processed

;; For stereo
(arrayp s)                 ; Check if stereo
(aref s 0)                 ; Left channel
(aref s 1)                 ; Right channel
```

### Time vs Samples

**Time-based** (preferred in Nyquist):
```lisp
(setf duration 1.0)  ; 1 second
```

**Sample-based**:
```lisp
(setf samples 44100)  ; Samples at 44.1kHz
(setf duration (/ samples *sound-srate*))  ; Convert to time
```

**Sample rate**:
```lisp
*sound-srate*  ; Current sample rate (e.g., 44100)
```

### Signal Algebra

Nyquist uses mathematical operations on entire signals:

```lisp
;; Addition
(sum signal1 signal2)
(+ signal1 signal2)  ; Shorthand

;; Multiplication
(mult signal1 signal2)
(* signal1 signal2)  ; Shorthand

;; Scaling
(mult signal 0.5)  ; Reduce amplitude by 50%
(scale 0.5 signal) ; Same thing

;; Inversion
(mult signal -1)

;; Offset
(sum signal 0.1)  ; Add DC offset
```

---

## Available Functions

### Filtering

**Low-pass filters**:
```lisp
(lowpass2 signal cutoff-hz)              ; 2nd order (12 dB/octave)
(lowpass4 signal cutoff-hz)              ; 4th order (24 dB/octave)
(lowpass6 signal cutoff-hz)              ; 6th order (36 dB/octave)
(lowpass8 signal cutoff-hz)              ; 8th order (48 dB/octave)
```

**High-pass filters**:
```lisp
(highpass2 signal cutoff-hz)
(highpass4 signal cutoff-hz)
(highpass6 signal cutoff-hz)
(highpass8 signal cutoff-hz)
```

**Band-pass filters**:
```lisp
(bandpass2 signal center-hz bandwidth-hz)
(bandpass4 signal center-hz bandwidth-hz)
```

**Notch filters**:
```lisp
(notch2 signal center-hz bandwidth-hz)
```

**Example**:
```lisp
;; Remove frequencies below 100 Hz
(setf filtered (highpass2 *track* 100))

;; Remove 60 Hz hum
(setf clean (notch2 *track* 60 10))
```

### Signal Analysis

**Absolute value**:
```lisp
(snd-abs signal)
```

**Square root**:
```lisp
(snd-sqrt signal)
```

**Average (moving average)**:
```lisp
(snd-avg signal window-size step-size operation)

;; Example: RMS calculation
(setf squared (mult signal signal))
(setf avg (snd-avg squared 100 100 op-average))
(setf rms (snd-sqrt avg))
```

**Maximum**:
```lisp
(snd-max signal1 signal2)  ; Element-wise maximum
(snd-maxv signal)           ; Peak value (returns number)
```

**Minimum**:
```lisp
(snd-min signal1 signal2)
```

### Signal Generation

**Sine wave**:
```lisp
(osc (hz-to-step frequency))        ; Sine wave
(osc (hz-to-step 440))              ; 440 Hz (A4)

;; With duration
(osc (hz-to-step 440) 1.0)          ; 1 second
```

**White noise**:
```lisp
(noise)                              ; White noise
(noise 1.0)                          ; 1 second of noise
```

**Silence**:
```lisp
(s-rest duration)
(s-rest 1.0)                         ; 1 second of silence
```

**DC signal** (constant):
```lisp
(const value duration)
(const 0.5 1.0)                      ; 0.5 amplitude for 1 second
```

**Ramp**:
```lisp
(ramp)                               ; 0 to 1 over track duration
(pwl time1 value1 time2 value2 ...)  ; Piecewise linear
```

### Envelope/Amplitude

**Envelope follower**:
```lisp
(snd-avg (snd-abs signal) window-size step-size op-peak)
```

**Peak detection**:
```lisp
(snd-maxv signal)  ; Returns peak value
```

**RMS calculation**:
```lisp
(defun calculate-rms (signal window-size)
  (let* ((squared (mult signal signal))
         (avg (snd-avg squared window-size window-size op-average)))
    (snd-sqrt avg)))
```

### Time Manipulation

**Stretch**:
```lisp
(stretch factor signal)
(stretch 2.0 signal)  ; Slow down by 2x
```

**Shift in time**:
```lisp
(at time signal)
(at 1.0 (noise 0.5))  ; Noise starts at 1 second
```

**Sequence**:
```lisp
(seq signal1 signal2 ...)
(seq (noise 0.5) (osc (hz-to-step 440) 0.5))  ; Noise then tone
```

**Simultaneous**:
```lisp
(sim signal1 signal2 ...)  ; Mix signals
```

### Comparison Operations

**Greater than**:
```lisp
(snd-greater signal threshold)  ; Returns 1 where signal > threshold, else 0
```

**Less than**:
```lisp
(snd-less signal threshold)
```

**Clipping**:
```lisp
;; Clip to range [-1, 1]
(snd-max (snd-min signal 1.0) -1.0)
```

---

## Limitations and Workarounds

### Limitation 1: No Direct Sample Access

**Problem**: Can't access individual samples like `audio[i]`.

**Impact**: Can't implement:
- Sample-by-sample interpolation
- Precise click location detection
- Array-based algorithms

**Workaround**: Use signal-level operations instead.

**Example**:
```lisp
;; Can't do: audio[100:200] = interpolate(...)
;; Instead: Use signal algebra and windowing

(setf processed (my-signal-level-function signal))
```

---

### Limitation 2: No Arrays/Lists of Samples

**Problem**: Can't create sample arrays like Python's NumPy.

**Impact**: Can't implement:
- AR linear prediction (requires matrix operations)
- FFT-based processing (no direct FFT access)
- Sophisticated statistical analysis

**Workaround**: Use built-in spectral functions or approximate with filters.

**Example**:
```lisp
;; Can't compute autocorrelation directly
;; Instead: Use filtering and signal operations
```

---

### Limitation 3: Limited Control Flow

**Problem**: Nyquist is functional, not imperative.

**Impact**:
- No traditional `for` loops over samples
- No `while` loops
- Limited `if-then-else` for signals

**Workaround**: Use recursion and functional programming patterns.

**Example**:
```lisp
;; Can't do:
;; for i in range(len(signal)):
;;     if signal[i] > threshold:
;;         signal[i] = 0

;; Instead: Use signal operations
(setf masked (snd-greater signal threshold))
(setf processed (mult signal (sum 1 (mult masked -1))))
```

---

### Limitation 4: No Variable-Length Windows

**Problem**: Window sizes must be constants, can't vary per location.

**Impact**: Difficult to implement adaptive algorithms.

**Workaround**: Use multiple passes with different window sizes.

---

### Limitation 5: Memory Constraints

**Problem**: Large buffers for snd-avg or complex operations can run out of memory.

**Impact**: May crash on long audio files.

**Workaround**:
- Use smaller window sizes
- Process in chunks if possible
- Optimize algorithm

---

### Limitation 6: No FFT/Spectral Editing

**Problem**: No direct access to FFT or spectral domain.

**Impact**: Can't implement:
- Spectral click removal
- Frequency-domain interpolation

**Workaround**: Use filterbanks (multiple bandpass filters).

---

## Best Practices

### 1. Signal-Level Thinking

**Think in transformations, not samples**:

```lisp
;; Good (signal-level)
(setf filtered (lowpass2 signal 1000))
(setf amplified (mult filtered 2.0))

;; Can't do (sample-level - doesn't exist in Nyquist)
;; for i in range(len(signal)):
;;     output[i] = process(signal[i])
```

### 2. Use Built-in Functions

Nyquist's built-in functions are optimized. Use them when possible.

```lisp
;; Good
(setf filtered (lowpass8 signal 2000))

;; Bad (trying to implement your own lowpass)
;; Complex LISP code that's slower and less accurate
```

### 3. Functional Composition

Build complex effects by composing simple functions:

```lisp
(defun my-effect (signal)
  (let* ((stage1 (highpass2 signal 100))
         (stage2 (lowpass8 stage1 8000))
         (stage3 (mult stage2 0.8)))
    stage3))
```

### 4. Use `let*` for Intermediate Results

```lisp
(let* ((filtered (lowpass2 signal 1000))
       (envelope (snd-abs filtered))
       (rms (snd-sqrt (mult envelope envelope))))
  rms)
```

`let*` allows later bindings to reference earlier ones.

### 5. Stereo Processing

Always check if audio is stereo:

```lisp
(cond
  ((arrayp s)
   ;; Stereo
   (vector
     (process-mono (aref s 0))
     (process-mono (aref s 1))))
  (t
   ;; Mono
   (process-mono s)))
```

### 6. Error Handling

Check for edge cases:

```lisp
(if (< *sound-srate* 8000)
    (format nil "Error: Sample rate too low")
    (process-audio))
```

### 7. Comments and Documentation

Document your code thoroughly:

```lisp
;; Calculate RMS with 10ms window
;; Window size in samples
(setf window-size (truncate (* 0.01 *sound-srate*)))

;; Compute moving average of squared signal
(setf avg (snd-avg squared window-size window-size op-average))
```

---

## Example Implementations

### Example 1: Simple Low-Pass Filter Effect

```lisp
;nyquist plug-in
;version 4
;type process
;name "Simple Low-Pass"
;control cutoff "Cutoff Frequency (Hz)" real "1000" 20 20000

(lowpass8 *track* cutoff)
```

### Example 2: Volume Adjustment with Clipping

```lisp
;nyquist plug-in
;version 4
;type process
;name "Safe Amplify"
;control gain-db "Gain (dB)" real "0" -20 20

;; Convert dB to linear gain
(setf gain (db-to-linear gain-db))

;; Amplify
(setf amplified (mult *track* gain))

;; Clip to prevent distortion
(setf clipped (snd-max (snd-min amplified 1.0) -1.0))

clipped
```

### Example 3: Click Removal (Frequency Domain Approach)

```lisp
;nyquist plug-in
;version 4
;type process
;preview enabled
;name "Click Removal"
;control threshold "Sensitivity" int "20" 1 100
;control max-width "Max Width (ms)" real "2.0" 0.1 10.0

;; Convert parameters
(setf threshold-factor (/ threshold 10.0))
(setf window-size (truncate (* 0.01 *sound-srate*)))

;; Function to process mono audio
(defun remove-clicks (sound)
  (let* (
    ;; Separate frequency bands
    (low-freq (lowpass8 sound 2000))
    (high-freq (highpass8 sound 2000))

    ;; Calculate envelope of high-freq component
    (hf-abs (snd-abs high-freq))
    (hf-envelope (snd-avg hf-abs window-size window-size op-peak))

    ;; Create threshold signal
    (threshold-signal (mult hf-envelope threshold-factor))

    ;; Detect clicks (where signal exceeds threshold)
    (clicks (snd-greater hf-abs threshold-signal))

    ;; Attenuate clicks (reduce to 30% where detected)
    (attenuation (sum 0.3 (mult clicks -0.7)))
    (hf-processed (mult high-freq attenuation))

    ;; Recombine
    (result (sum low-freq hf-processed)))
    result))

;; Process stereo or mono
(cond
  ((arrayp s)
   (vector
     (remove-clicks (aref s 0))
     (remove-clicks (aref s 1))))
  (t
   (remove-clicks s)))
```

### Example 4: RMS Normalization

```lisp
;nyquist plug-in
;version 4
;type process
;name "RMS Normalize"
;control target-db "Target RMS (dB)" real "-20" -40 0

;; Calculate RMS
(defun calc-rms (signal)
  (let* ((squared (mult signal signal))
         (duration (/ (snd-length signal ny:all) *sound-srate*))
         (avg (snd-avg squared
                      (truncate *sound-srate*)
                      (truncate *sound-srate*)
                      op-average)))
    (snd-sqrt avg)))

;; Process
(let* ((current-rms (calc-rms *track*))
       (target-linear (db-to-linear target-db))
       (current-db (linear-to-db (snd-maxv current-rms)))
       (gain-db (- target-db current-db))
       (gain (db-to-linear gain-db)))
  (mult *track* gain))
```

### Example 5: Multi-Pass Processing

```lisp
;nyquist plug-in
;version 4
;type process
;name "Multi-Pass Click Removal"
;control passes "Number of Passes" int "2" 1 5
;control sensitivity "Sensitivity" int "20" 1 100

(defun process-once (sound thresh)
  (let* ((low (lowpass8 sound 2000))
         (high (highpass8 sound 2000))
         (high-env (snd-avg (snd-abs high) 100 100 op-peak))
         (threshold-sig (mult high-env thresh))
         (attenuation (snd-min (/ threshold-sig (snd-abs high)) 1.0))
         (high-proc (mult high attenuation)))
    (sum low high-proc)))

(defun multi-pass (sound passes thresh)
  (if (<= passes 0)
      sound
      (multi-pass
        (process-once sound thresh)
        (- passes 1)
        (* thresh 0.8))))

(multi-pass *track* passes (/ sensitivity 10.0))
```

---

## Debugging Tips

### 1. Use Nyquist Prompt

**Access**: `Tools > Nyquist Prompt` in Audacity

Test code snippets without creating a plugin file:

```lisp
;; Test in Nyquist Prompt
(lowpass2 s 1000)
```

### 2. Enable Debug Flags

```lisp
;debugflags trace
```

This shows execution trace in Audacity's debug log.

### 3. Return Intermediate Values

Return intermediate results to inspect:

```lisp
;; Instead of final result
;; (process-audio s)

;; Return intermediate value
(setf intermediate (highpass2 s 100))
intermediate  ; Return this to hear/see it
```

### 4. Use `format` for Text Output

For analyze-type plugins:

```lisp
;type analyze
(format nil "Peak: ~a~%" (snd-maxv s))
```

### 5. Check Sample Rate

```lisp
(if (< *sound-srate* 22050)
    (format nil "Error: Sample rate too low: ~a" *sound-srate*)
    (process-audio))
```

### 6. Validate Parameters

```lisp
(if (> threshold 100)
    (setf threshold 100))
(if (< threshold 1)
    (setf threshold 1))
```

### 7. Catch Errors

```lisp
(if (not (soundp s))
    (format nil "Error: No audio selected")
    (process-audio s))
```

---

## Advanced Techniques

### Custom Function Definition

```lisp
(defun my-function (param1 param2)
  (let* ((intermediate (mult param1 param2)))
    intermediate))

;; Use it
(my-function *track* 0.5)
```

### Recursive Processing

```lisp
(defun recursive-process (sound count)
  (if (<= count 0)
      sound
      (recursive-process
        (lowpass2 sound 1000)
        (- count 1))))

;; Apply lowpass 5 times
(recursive-process *track* 5)
```

### Conditional Processing

```lisp
(cond
  ((< *sound-srate* 22050)
   (format nil "Sample rate too low"))
  ((> (snd-length s ny:all) 10000000)
   (format nil "Audio too long"))
  (t
   (process-audio s)))
```

### Windowing Functions

```lisp
;; Create a window
(defun hann-window (length)
  (let* ((half-pi (/ 3.141592653589793 2.0)))
    (mult
      (osc (hz-to-step (/ 1.0 length)) length)
      (osc (hz-to-step (/ 1.0 length)) length))))

;; Apply to signal
(mult signal (hann-window duration))
```

---

## Common Patterns

### Pattern 1: Split-Process-Combine

```lisp
(let* ((low (lowpass2 s 500))
       (mid (bandpass2 s 1000 1000))
       (high (highpass2 s 2000))
       ;; Process each band
       (low-proc (mult low 1.2))
       (mid-proc mid)
       (high-proc (mult high 0.8)))
  ;; Combine
  (sum low-proc mid-proc high-proc))
```

### Pattern 2: Envelope Following

```lisp
(let* ((abs-signal (snd-abs s))
       (envelope (snd-avg abs-signal 1000 1000 op-peak)))
  envelope)
```

### Pattern 3: Dynamic Range Compression

```lisp
(let* ((envelope (snd-avg (snd-abs s) 100 100 op-peak))
       (threshold 0.5)
       (over-threshold (snd-greater envelope threshold))
       (gain-reduction (mult over-threshold -0.5))
       (total-gain (sum 1.0 gain-reduction)))
  (mult s total-gain))
```

---

## Performance Considerations

### 1. Window Sizes

Smaller windows = faster processing, less smooth results
Larger windows = slower processing, smoother results

```lisp
;; Fast but choppy
(snd-avg signal 10 10 op-average)

;; Slow but smooth
(snd-avg signal 1000 1000 op-average)
```

### 2. Filter Orders

Higher order filters are more expensive:

```lisp
(lowpass2 s 1000)  ; Fast
(lowpass8 s 1000)  ; Slower, but sharper cutoff
```

### 3. Avoid Excessive Nesting

```lisp
;; Bad (deeply nested)
(mult (sum (lowpass2 (highpass2 s 100) 1000) 0.5) 2.0)

;; Good (use let*)
(let* ((stage1 (highpass2 s 100))
       (stage2 (lowpass2 stage1 1000))
       (stage3 (sum stage2 0.5))
       (stage4 (mult stage3 2.0)))
  stage4)
```

---

## Resources and Further Reading

### Official Documentation
- **Nyquist Reference Manual**: https://www.cs.cmu.edu/~rbd/doc/nyquist/
- **Audacity Nyquist Tutorial**: https://manual.audacityteam.org/man/nyquist.html
- **Plugin Examples**: https://plugins.audacityteam.org/

### Example Plugins
Look at Audacity's included Nyquist plugins:
- Click Removal
- Noise Reduction
- Equalizer
- Reverb

### LISP Resources
- "Practical Common Lisp" by Peter Seibel (free online)
- "The Little Schemer" (similar language)

### Audio DSP
- "The Scientist and Engineer's Guide to Digital Signal Processing"
- "DAFX: Digital Audio Effects" by Udo Zölzer

---

## Conclusion

Nyquist is a powerful but specialized language for audio processing. Its signal-level approach is elegant for certain tasks (filtering, mixing, synthesis) but limiting for others (sample-level interpolation, statistical analysis).

**Best suited for**:
- Filtering effects
- Dynamic processing
- Synthesis
- Simple audio transformations

**Not ideal for**:
- Complex click removal with interpolation
- FFT-based processing
- Machine learning
- Precise sample-level control

For vinyl scratch removal specifically, Nyquist can implement **adaptive attenuation** approaches effectively, but true **interpolation-based** methods require sample-level access better suited to Python/C++.

---

*Last updated: 2025-11-07*
*Compiled by: Claude (Anthropic)*
