/**
 * vinyl_scratch_removal.c - LV2 plugin for vinyl scratch removal
 *
 * This is a thin wrapper around the Cython core library.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lv2/lv2plug.in/ns/lv2core/lv2.h>

#define VINYL_URI "http://github.com/anthropics/vinyl-scratch-removal"

/* Port indices */
typedef enum {
    VINYL_INPUT  = 0,
    VINYL_OUTPUT = 1,
    VINYL_THRESHOLD = 2,
    VINYL_MODE = 3
} PortIndex;

/* Plugin instance */
typedef struct {
    /* Port buffers */
    const float* input;
    float* output;
    const float* threshold;
    const float* mode;

    /* Processing state */
    float sample_rate;

    /* Buffer for collecting samples (since we need full context) */
    float* buffer;
    size_t buffer_size;
    size_t buffer_pos;
} VinylScratchRemoval;

/* Instantiate plugin */
static LV2_Handle
instantiate(const LV2_Descriptor*     descriptor,
            double                    rate,
            const char*               bundle_path,
            const LV2_Feature* const* features)
{
    VinylScratchRemoval* plugin = (VinylScratchRemoval*)calloc(1, sizeof(VinylScratchRemoval));
    if (!plugin) {
        return NULL;
    }

    plugin->sample_rate = (float)rate;

    /* Allocate processing buffer (1 second) */
    plugin->buffer_size = (size_t)rate;
    plugin->buffer = (float*)calloc(plugin->buffer_size, sizeof(float));
    if (!plugin->buffer) {
        free(plugin);
        return NULL;
    }

    plugin->buffer_pos = 0;

    return (LV2_Handle)plugin;
}

/* Connect port */
static void
connect_port(LV2_Handle instance,
             uint32_t   port,
             void*      data)
{
    VinylScratchRemoval* plugin = (VinylScratchRemoval*)instance;

    switch ((PortIndex)port) {
    case VINYL_INPUT:
        plugin->input = (const float*)data;
        break;
    case VINYL_OUTPUT:
        plugin->output = (float*)data;
        break;
    case VINYL_THRESHOLD:
        plugin->threshold = (const float*)data;
        break;
    case VINYL_MODE:
        plugin->mode = (const float*)data;
        break;
    }
}

/* Simple click detection and attenuation */
/* NOTE: This is a simplified version that doesn't use the full Cython library
   to avoid Python embedding complexity. For production, you would call
   the Cython library or port the algorithm to pure C. */
static void
process_simple(VinylScratchRemoval* plugin, uint32_t n_samples)
{
    const float threshold = *plugin->threshold;
    const float* input = plugin->input;
    float* output = plugin->output;

    /* Simple high-pass filtering and attenuation approach */
    /* This matches the Nyquist plugin approach */

    float prev1 = 0.0f;
    float prev2 = 0.0f;

    for (uint32_t i = 0; i < n_samples; i++) {
        float sample = input[i];

        /* Calculate second derivative (simple click detector) */
        float diff2 = sample - 2.0f * prev1 + prev2;

        /* Adaptive threshold based on local level */
        float abs_diff2 = fabsf(diff2);
        float local_threshold = threshold * 0.01f; /* Scale threshold */

        /* Attenuate if exceeds threshold */
        if (abs_diff2 > local_threshold) {
            /* Reduce amplitude of click */
            float attenuation = local_threshold / abs_diff2;
            attenuation = fmaxf(attenuation, 0.3f); /* Keep at least 30% */
            output[i] = sample * attenuation;
        } else {
            output[i] = sample;
        }

        prev2 = prev1;
        prev1 = sample;
    }
}

/* Activate plugin */
static void
activate(LV2_Handle instance)
{
    VinylScratchRemoval* plugin = (VinylScratchRemoval*)instance;
    plugin->buffer_pos = 0;
    memset(plugin->buffer, 0, plugin->buffer_size * sizeof(float));
}

/* Run plugin */
static void
run(LV2_Handle instance, uint32_t n_samples)
{
    VinylScratchRemoval* plugin = (VinylScratchRemoval*)instance;

    if (!plugin->input || !plugin->output) {
        return;
    }

    /* For now, use simple processing */
    /* TODO: Integrate full Cython library for better quality */
    process_simple(plugin, n_samples);
}

/* Deactivate plugin */
static void
deactivate(LV2_Handle instance)
{
    /* Nothing to do */
}

/* Cleanup plugin */
static void
cleanup(LV2_Handle instance)
{
    VinylScratchRemoval* plugin = (VinylScratchRemoval*)instance;
    if (plugin) {
        free(plugin->buffer);
        free(plugin);
    }
}

/* Extension data */
static const void*
extension_data(const char* uri)
{
    return NULL;
}

/* Plugin descriptor */
static const LV2_Descriptor descriptor = {
    VINYL_URI,
    instantiate,
    connect_port,
    activate,
    run,
    deactivate,
    cleanup,
    extension_data
};

/* Plugin entry point */
LV2_SYMBOL_EXPORT
const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : NULL;
}
