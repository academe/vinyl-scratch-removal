#!/usr/bin/env python3
"""
Test script for vinyl scratch removal core library.
"""

import sys
import os
import numpy as np

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from vinyl_core import VinylProcessor, MODE_STANDARD
    print("✓ Successfully imported vinyl_core")
except ImportError as e:
    print(f"✗ Failed to import vinyl_core: {e}")
    print("\nPlease build the core library first:")
    print("  cd core && python setup.py build_ext --inplace")
    sys.exit(1)


def test_basic_processing():
    """Test basic click detection and removal."""
    print("\nTest 1: Basic processing")
    print("-" * 40)

    # Create test audio with a click
    sample_rate = 44100
    duration = 1.0  # 1 second
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

    # Add a click at sample 1000
    click_pos = 1000
    audio[click_pos:click_pos + 5] = 1.0

    print(f"  Original peak at click: {audio[click_pos:click_pos + 5].max():.3f}")

    # Process
    processor = VinylProcessor(sample_rate, threshold=3.0)
    processed = processor.process(audio.copy())

    print(f"  Processed peak at click: {processed[click_pos:click_pos + 5].max():.3f}")

    # Verify click was reduced
    original_peak = audio[click_pos:click_pos + 5].max()
    processed_peak = processed[click_pos:click_pos + 5].max()

    if processed_peak < original_peak * 0.5:
        print("  ✓ Click successfully reduced")
        return True
    else:
        print("  ✗ Click not adequately reduced")
        return False


def test_detection():
    """Test click detection."""
    print("\nTest 2: Click detection")
    print("-" * 40)

    # Create test audio with multiple clicks
    sample_rate = 44100
    audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

    # Add clicks at known positions
    click_positions = [1000, 5000, 10000]
    for pos in click_positions:
        audio[pos:pos + 3] = 1.0

    print(f"  Added {len(click_positions)} synthetic clicks")

    # Detect
    processor = VinylProcessor(sample_rate, threshold=3.0)
    detected = processor.detect(audio)

    print(f"  Detected {len(detected)} clicks")

    # Verify detection
    if len(detected) >= len(click_positions):
        print("  ✓ Detection working correctly")
        return True
    else:
        print("  ✗ Not all clicks detected")
        return False


def test_modes():
    """Test different detection modes."""
    print("\nTest 3: Detection modes")
    print("-" * 40)

    # Create test audio
    sample_rate = 44100
    audio = np.random.randn(10000).astype(np.float32) * 0.1
    audio[1000:1005] = 0.5  # Medium click

    # Test each mode
    modes = [
        (0, "Conservative"),
        (1, "Standard"),
        (2, "Aggressive")
    ]

    results = {}
    for mode_id, mode_name in modes:
        processor = VinylProcessor(sample_rate, threshold=3.0, mode=mode_id)
        detected = processor.detect(audio)
        results[mode_name] = len(detected)
        print(f"  {mode_name:12s}: {len(detected)} clicks detected")

    # Aggressive should detect most, conservative least
    if results["Aggressive"] >= results["Conservative"]:
        print("  ✓ Mode behavior correct")
        return True
    else:
        print("  ✗ Mode behavior unexpected")
        return False


def test_file_processing():
    """Test file-based processing."""
    print("\nTest 4: File processing")
    print("-" * 40)

    try:
        import soundfile as sf
    except ImportError:
        print("  ⚠ Skipping (soundfile not installed)")
        return None

    # Create test file
    sample_rate = 44100
    audio = np.random.randn(sample_rate).astype(np.float32) * 0.1
    audio[1000:1005] = 1.0

    test_input = "/tmp/test_vinyl_input.wav"
    test_output = "/tmp/test_vinyl_output.wav"

    try:
        # Write test file
        sf.write(test_input, audio, sample_rate)
        print(f"  Created test file: {test_input}")

        # Process
        processor = VinylProcessor(sample_rate, threshold=3.0)
        processor.process_file(test_input, test_output)

        # Verify output exists
        if os.path.exists(test_output):
            output, sr = sf.read(test_output)
            print(f"  Output file created: {test_output}")
            print(f"  ✓ File processing working")
            return True
        else:
            print("  ✗ Output file not created")
            return False

    finally:
        # Cleanup
        for f in [test_input, test_output]:
            if os.path.exists(f):
                os.remove(f)


def main():
    """Run all tests."""
    print("=" * 50)
    print("Vinyl Scratch Removal - Core Library Tests")
    print("=" * 50)

    tests = [
        test_basic_processing,
        test_detection,
        test_modes,
        test_file_processing,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⚠ SKIP"
        print(f"  {status:10s} {name}")

    print(f"\nPassed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        print("\n⚠ Some tests failed!")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
