# cython: language_level=3
# interpolation.pyx - AR and cubic spline interpolation using Cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from scipy.linalg.cython_lapack cimport sposv

ctypedef np.float32_t DTYPE_t

def interpolate_cubic_python(np.ndarray[DTYPE_t, ndim=1] audio,
                             int start,
                             int end):
    """
    Cubic Hermite spline interpolation.

    Python-friendly interface.
    """
    cdef int context = 4
    cdef int length = len(audio)
    cdef int gap_len = end - start
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(gap_len, dtype=np.float32)

    if start < context or end + context >= length:
        # Not enough context, use linear interpolation
        if start > 0 and end < length:
            cdef float slope = (audio[end] - audio[start-1]) / (end - start + 1)
            cdef int i
            for i in range(gap_len):
                result[i] = audio[start-1] + slope * (i + 1)
        return result

    # Call C-level function
    interpolate_cubic_internal(
        &audio[0], length, start, end, &result[0]
    )

    return result


cdef void interpolate_cubic_internal(float* audio,
                                     int length,
                                     int start,
                                     int end,
                                     float* output) nogil:
    """
    C-level cubic Hermite interpolation (no Python objects, no GIL).
    """
    cdef int context = 4
    cdef int gap_len = end - start
    cdef int i
    cdef float t, t2, t3
    cdef float h00, h10, h01, h11
    cdef float p0, p1, m0, m1

    # Use samples around the gap
    # Context points: start-context to start, end to end+context
    # For simplicity, use start-1 and end as control points

    if start > 0 and end < length:
        p0 = audio[start - 1]
        p1 = audio[end]

        # Calculate tangents (simple finite differences)
        if start > 1:
            m0 = (audio[start] - audio[start - 2]) / 2.0
        else:
            m0 = audio[start] - audio[start - 1]

        if end < length - 1:
            m1 = (audio[end + 1] - audio[end - 1]) / 2.0
        else:
            m1 = audio[end] - audio[end - 1]

        # Hermite interpolation
        for i in range(gap_len):
            t = <float>(i + 1) / <float>(gap_len + 1)
            t2 = t * t
            t3 = t2 * t

            # Hermite basis functions
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2

            output[i] = h00*p0 + h10*m0 + h01*p1 + h11*m1
    else:
        # Fallback: zeros
        for i in range(gap_len):
            output[i] = 0.0


def interpolate_ar_python(np.ndarray[DTYPE_t, ndim=1] audio,
                         int start,
                         int end,
                         int ar_order=20):
    """
    Autoregressive linear prediction interpolation.

    Python-friendly interface using NumPy for matrix operations.
    """
    cdef int context_len = ar_order * 4
    cdef int length = len(audio)
    cdef int gap_len = end - start

    # Check if we have enough context
    if start < context_len or end + context_len >= length:
        # Fall back to cubic interpolation
        return interpolate_cubic_python(audio, start, end)

    # Extract context samples
    cdef np.ndarray[DTYPE_t, ndim=1] before = audio[start - context_len:start]
    cdef np.ndarray[DTYPE_t, ndim=1] after = audio[end:end + context_len]
    cdef np.ndarray[DTYPE_t, ndim=1] context = np.concatenate([before, after])

    # Calculate autocorrelation
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.correlate(context, context, mode='full')
    cdef int mid = len(r) // 2
    r = r[mid:mid + ar_order + 1]

    # Solve Yule-Walker equations: R * a = r
    # Using scipy for matrix inversion (pure Python for now, can optimize later)
    try:
        from scipy.linalg import solve_toeplitz
        cdef np.ndarray[DTYPE_t, ndim=1] ar_coeffs = solve_toeplitz(
            r[:ar_order].astype(np.float32),
            r[1:ar_order + 1].astype(np.float32)
        ).astype(np.float32)

        # Forward prediction
        cdef np.ndarray[DTYPE_t, ndim=1] interpolated = np.zeros(gap_len, dtype=np.float32)
        cdef int i, k
        cdef float prediction

        for i in range(gap_len):
            prediction = 0.0

            # Use samples before the gap and already predicted samples
            for k in range(min(i + 1, ar_order)):
                if i - k - 1 >= 0:
                    # Use predicted samples
                    prediction += ar_coeffs[k] * interpolated[i - k - 1]
                else:
                    # Use actual samples before gap
                    cdef int idx = start - 1 - k + i
                    if idx >= 0:
                        prediction += ar_coeffs[k] * audio[idx]

            interpolated[i] = prediction

        return interpolated

    except:
        # Fall back to cubic interpolation on error
        return interpolate_cubic_python(audio, start, end)


def blend_interpolation(np.ndarray[DTYPE_t, ndim=1] interpolated):
    """
    Create Tukey window for smooth blending.
    """
    cdef int length = len(interpolated)
    cdef np.ndarray[DTYPE_t, ndim=1] window

    if length < 4:
        return np.ones(length, dtype=np.float32)

    # Simple linear taper on edges
    window = np.ones(length, dtype=np.float32)
    cdef int taper = min(length // 4, 10)
    cdef int i
    cdef float t

    for i in range(taper):
        t = <float>i / <float>taper
        window[i] = t
        window[length - 1 - i] = t

    return window
