# cython: language_level=3
# detection.pyx - Click detection using Cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free

# Define numpy types for Cython
ctypedef np.float32_t DTYPE_t

cdef struct Click:
    int start
    int end

def detect_clicks_python(np.ndarray[DTYPE_t, ndim=1] audio,
                         float threshold,
                         int max_width_samples,
                         int window_size):
    """
    Detect clicks using second derivative analysis.

    Python-friendly interface that returns list of (start, end) tuples.
    """
    cdef list result = []
    cdef int i, length = len(audio)
    cdef int num_clicks
    cdef Click* clicks

    # Call C-level function
    clicks = detect_clicks_internal(
        &audio[0], length, threshold, max_width_samples, window_size, &num_clicks
    )

    # Convert to Python list
    for i in range(num_clicks):
        result.append((clicks[i].start, clicks[i].end))

    if clicks != NULL:
        free(clicks)

    return result


cdef Click* detect_clicks_internal(float* audio,
                                   int length,
                                   float threshold,
                                   int max_width,
                                   int window_size,
                                   int* num_clicks) nogil:
    """
    C-level click detection function (no Python objects, no GIL).

    Returns dynamically allocated array of Click structs.
    Caller must free the returned array.
    """
    cdef float* diff2
    cdef float* local_rms
    cdef int i, j, count
    cdef float rms_sum
    cdef Click* clicks
    cdef int max_clicks = 10000
    cdef int click_count = 0
    cdef int in_click = 0
    cdef int click_start = 0

    # Allocate memory for second derivative
    diff2 = <float*>malloc(length * sizeof(float))
    if diff2 == NULL:
        num_clicks[0] = 0
        return NULL

    # Calculate second derivative
    for i in range(2, length):
        diff2[i] = audio[i] - 2*audio[i-1] + audio[i-2]

    diff2[0] = 0
    diff2[1] = 0

    # Calculate local RMS
    local_rms = <float*>malloc(length * sizeof(float))
    if local_rms == NULL:
        free(diff2)
        num_clicks[0] = 0
        return NULL

    for i in range(length):
        rms_sum = 0.0
        count = 0

        # Calculate RMS in window
        for j in range(max(0, i - window_size), min(length, i + window_size)):
            rms_sum += audio[j] * audio[j]
            count += 1

        if count > 0:
            local_rms[i] = sqrt(rms_sum / count)
        else:
            local_rms[i] = 0.0

    # Allocate click array
    clicks = <Click*>malloc(max_clicks * sizeof(Click))
    if clicks == NULL:
        free(diff2)
        free(local_rms)
        num_clicks[0] = 0
        return NULL

    # Detect clicks
    for i in range(2, length):
        cdef float abs_diff2 = fabs(diff2[i])
        cdef float thresh = local_rms[i] * threshold

        if abs_diff2 > thresh:
            if not in_click:
                click_start = i
                in_click = 1
        else:
            if in_click:
                # End of click
                cdef int width = i - click_start
                if width > 0 and width <= max_width:
                    if click_count < max_clicks:
                        clicks[click_count].start = click_start
                        clicks[click_count].end = i
                        click_count += 1
                in_click = 0

    # Handle click at end of audio
    if in_click:
        cdef int width = length - click_start
        if width > 0 and width <= max_width and click_count < max_clicks:
            clicks[click_count].start = click_start
            clicks[click_count].end = length
            click_count += 1

    free(diff2)
    free(local_rms)

    num_clicks[0] = click_count
    return clicks


def calculate_rms(np.ndarray[DTYPE_t, ndim=1] audio, int window_size):
    """
    Calculate local RMS with sliding window.
    """
    cdef int i, j, length = len(audio)
    cdef np.ndarray[DTYPE_t, ndim=1] rms = np.zeros(length, dtype=np.float32)
    cdef float rms_sum
    cdef int count

    for i in range(length):
        rms_sum = 0.0
        count = 0

        for j in range(max(0, i - window_size), min(length, i + window_size)):
            rms_sum += audio[j] * audio[j]
            count += 1

        if count > 0:
            rms[i] = sqrt(rms_sum / count)

    return rms
