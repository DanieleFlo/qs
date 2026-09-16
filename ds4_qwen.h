#ifndef DS4_QWEN_H
#define DS4_QWEN_H

#include <stdbool.h>
#include <stdint.h>

/* Execution phases shared by the Qwen graph scheduler and GPU backends.
 * They describe why a graph operation is running; row count and tensor shape
 * remain the authoritative kernel-dispatch inputs. */
typedef enum {
    DS4_QWEN_STAGE_DECODE = 0,
    DS4_QWEN_STAGE_PREFILL,
    DS4_QWEN_STAGE_MTP_VERIFY,
    DS4_QWEN_STAGE_MTP_DRAFT,
    DS4_QWEN_STAGE_MTP_REPLAY,
    DS4_QWEN_STAGE_MTP_CATCHUP,
} ds4_qwen_execution_stage;

/* Passed by value per projection. Neither field replaces shape/row checks;
 * concurrent schedulers must not communicate through a backend global. */
typedef struct {
    ds4_qwen_execution_stage stage;
    uint32_t layer;
} ds4_qwen_execution_context;

/* Output consumers are independent: catch-up needs hidden on device,
 * a device consumer may need logits, and only a host consumer needs readback.
 * Close the dependencies here so a new caller cannot omit an earlier stage. */
typedef struct {
    bool hidden;
    bool logits;
    bool read_logits;
} ds4_qwen_output_plan;

static inline ds4_qwen_output_plan ds4_qwen_plan_outputs(
        bool hidden, bool logits, bool read_logits) {
    ds4_qwen_output_plan plan = {
        hidden || logits || read_logits, logits || read_logits, read_logits
    };
    return plan;
}

static inline const char *ds4_qwen_execution_stage_name(
        ds4_qwen_execution_stage stage) {
    switch (stage) {
    case DS4_QWEN_STAGE_DECODE: return "decode";
    case DS4_QWEN_STAGE_PREFILL: return "prefill";
    case DS4_QWEN_STAGE_MTP_VERIFY: return "mtp-verify";
    case DS4_QWEN_STAGE_MTP_DRAFT: return "mtp-draft";
    case DS4_QWEN_STAGE_MTP_REPLAY: return "mtp-replay";
    case DS4_QWEN_STAGE_MTP_CATCHUP: return "mtp-catchup";
    }
    return "unknown";
}

#endif
