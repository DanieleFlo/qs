#ifndef DS4_QWEN_H
#define DS4_QWEN_H

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
