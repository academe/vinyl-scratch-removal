/**
 * vinyl_core.h - C API for vinyl scratch removal library
 *
 * This provides a C-compatible interface to the Cython-based
 * vinyl scratch removal core library.
 */

#ifndef VINYL_CORE_H
#define VINYL_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Opaque processor handle */
typedef struct VinylProcessorHandle VinylProcessorHandle;

/* Detection modes */
typedef enum {
    VINYL_MODE_CONSERVATIVE = 0,
    VINYL_MODE_STANDARD = 1,
    VINYL_MODE_AGGRESSIVE = 2
} VinylMode;

/* Configuration structure */
typedef struct {
    float sample_rate;      /* Sample rate in Hz */
    float threshold;        /* Detection threshold (std deviations) */
    int ar_order;          /* AR model order */
    VinylMode mode;        /* Detection mode */
} VinylConfig;

/**
 * Create a new processor instance.
 *
 * @param config Configuration parameters
 * @return Processor handle, or NULL on error
 */
VinylProcessorHandle* vinyl_create(VinylConfig config);

/**
 * Destroy a processor instance.
 *
 * @param handle Processor handle
 */
void vinyl_destroy(VinylProcessorHandle* handle);

/**
 * Set detection threshold.
 *
 * @param handle Processor handle
 * @param threshold New threshold value
 */
void vinyl_set_threshold(VinylProcessorHandle* handle, float threshold);

/**
 * Set detection mode.
 *
 * @param handle Processor handle
 * @param mode Detection mode
 */
void vinyl_set_mode(VinylProcessorHandle* handle, VinylMode mode);

/**
 * Set maximum click width in milliseconds.
 *
 * @param handle Processor handle
 * @param width_ms Maximum click width in ms
 */
void vinyl_set_max_width(VinylProcessorHandle* handle, float width_ms);

/**
 * Process audio buffer (in-place).
 *
 * @param handle Processor handle
 * @param audio Audio buffer (float32, will be modified)
 * @param frames Number of frames
 * @return 0 on success, -1 on error
 */
int vinyl_process(VinylProcessorHandle* handle, float* audio, size_t frames);

/**
 * Process audio file.
 *
 * @param handle Processor handle
 * @param input_path Path to input WAV file
 * @param output_path Path to output WAV file
 * @return 0 on success, -1 on error
 */
int vinyl_process_file(VinylProcessorHandle* handle,
                       const char* input_path,
                       const char* output_path);

#ifdef __cplusplus
}
#endif

#endif /* VINYL_CORE_H */
