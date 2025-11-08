#!/usr/bin/env python3
"""
Command-line tool for vinyl scratch removal.

This is a Python wrapper around the Cython core library.
"""

import argparse
import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from vinyl_core import VinylProcessor, MODE_CONSERVATIVE, MODE_STANDARD, MODE_AGGRESSIVE
except ImportError:
    print("Error: vinyl_core module not found!")
    print("Please build the core library first:")
    print("  cd core && python setup.py build_ext --inplace")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Vinyl Scratch Removal - Remove clicks and pops from vinyl recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s input.wav output.wav

  # Aggressive mode
  %(prog)s input.wav output.wav --mode aggressive --threshold 2.0

  # Conservative mode
  %(prog)s input.wav output.wav --mode conservative --threshold 4.0

  # Custom AR order
  %(prog)s input.wav output.wav --ar-order 30

Detection Modes:
  conservative - Only remove obvious clicks
  standard     - Balanced detection (default)
  aggressive   - Maximum click removal (may affect audio)

Threshold:
  Lower values = more sensitive (detect more clicks)
  Higher values = less sensitive (detect fewer clicks)
  Typical range: 2.0 (aggressive) to 5.0 (conservative)
  Default: 3.0
        """
    )

    parser.add_argument('input', help='Input audio file (WAV format)')
    parser.add_argument('output', help='Output audio file (WAV format)')

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=3.0,
        help='Detection threshold (default: 3.0)'
    )

    parser.add_argument(
        '--max-width', '-w',
        type=float,
        default=2.0,
        help='Maximum click width in milliseconds (default: 2.0)'
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['conservative', 'standard', 'aggressive'],
        default='standard',
        help='Detection mode (default: standard)'
    )

    parser.add_argument(
        '--ar-order', '-a',
        type=int,
        default=20,
        help='AR model order for interpolation (default: 20, range: 10-50)'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Vinyl Scratch Removal 1.0.0'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.ar_order < 10 or args.ar_order > 50:
        print(f"Warning: AR order {args.ar_order} is outside typical range (10-50)")

    # Map mode to integer
    mode_map = {
        'conservative': MODE_CONSERVATIVE,
        'standard': MODE_STANDARD,
        'aggressive': MODE_AGGRESSIVE
    }
    mode = mode_map[args.mode]

    print(f"Vinyl Scratch Removal v1.0.0")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Threshold:  {args.threshold}")
    print(f"Max width:  {args.max_width} ms")
    print(f"Mode:       {args.mode}")
    print(f"AR order:   {args.ar_order}")
    print()

    try:
        # Create processor
        processor = VinylProcessor(
            threshold=args.threshold,
            ar_order=args.ar_order,
            mode=mode
        )

        processor.set_max_width_ms(args.max_width)

        # Process file
        print("Loading audio...")
        processor.process_file(args.input, args.output)

        print()
        print(f"Processed audio saved to: {args.output}")
        print("Done!")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
