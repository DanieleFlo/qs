#!/bin/sh
set -e

GLM_UNSLOTH_REPO="unsloth/GLM-5.2-GGUF"
GLM_ANTIREZ_REPO="antirez/GLM-5.2-GGUF"
QWEN36_REPO="ggml-org/Qwen3.6-27B-GGUF"
QWEN36_UNSLOTH_REPO="unsloth/Qwen3.6-27B-GGUF"
QWEN36_REVISION="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8"
REPO="antirez/deepseek-v4-gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
DSPARK_SUPPORT_FILE="DeepSeek-V4-Flash-DSpark-support.gguf"
GLM_UNSLOTH_Q4_REMOTE_BASE="UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_LOCAL_BASE="GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_FIRST_FILE="$GLM_UNSLOTH_Q4_LOCAL_BASE-00001-of-00011.gguf"
GLM_ANTIREZ_IQ2XXS_FILE="GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
GLM_ANTIREZ_Q2_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
GLM_ANTIREZ_Q4_FILE="GLM-5.2-UD-Q4_K_RoutedQ4K.gguf"
QWEN36_Q4_FILE="Qwen3.6-27B-Q4_K_M.gguf"
QWEN36_Q4_S_FILE="Qwen3.6-27B-Q4_K_S.gguf"
QWEN36_Q4_SHA256="65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921"
QWEN36_Q4_S_SHA256="ff857ba9f2184d8be408e8cabda12c89ba5adb202fddc1a88b3774d7bb232aca"
QWEN36_MTP_FILE="mtp-Qwen3.6-27B-Q4_0.gguf"
QWEN36_MTP_SHA256="3d593f9e2788d59bb30d6024706b1efd5219fea466b6397c46159e3540937173"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DwarfStar GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh pro-q2-imatrix [--token TOKEN]
  ./download_model.sh pro-q4-layers00-30 [--token TOKEN]
  ./download_model.sh pro-q4-layers31-output [--token TOKEN]
  ./download_model.sh pro-q4-split [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]
  ./download_model.sh dspark-support [--token TOKEN]
  ./download_model.sh glm-unsloth-q4 [--token TOKEN]
  ./download_model.sh glm-antirez-iq2xxs [--token TOKEN]
  ./download_model.sh glm-antirez-q2 [--token TOKEN]
  ./download_model.sh glm-antirez-q4 [--token TOKEN]
  ./download_model.sh qwen36-q4 [--token TOKEN]
  ./download_model.sh qwen36-mtp [--token TOKEN]
  ./download_model.sh qwen36-q4-mtp [--token TOKEN]

Targets:

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to q2-imatrix.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-q2-imatrix
       DeepSeek V4 PRO q2 imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  pro-q4-layers00-30
       First half of the DeepSeek V4 PRO Q4 routed-expert quant, layers 0..30.
       Use on the coordinator in a two-Mac-Studio distributed run. About 426 GB.

  pro-q4-layers31-output
       Second half of the DeepSeek V4 PRO Q4 routed-expert quant, layers
       31..output. Use on the worker in a two-Mac-Studio distributed run.
       About 412 GB.

  pro-q4-split
       Downloads both PRO Q4 split files into the download directory. About
       838 GB total. This target does not update ./ds4flash.gguf.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q2-q4-imatrix, and q4-imatrix, but must be
       enabled explicitly with --mtp-model FILE when running ds4 or ds4-server.

  dspark-support
       Optional DSpark speculative decoding support GGUF, about 6 GB. Enable it
       with --dspark and --mtp-model FILE when running ds4 or ds4-server.

  glm-unsloth-q4
       GLM 5.2 Unsloth UD-Q4_K_XL quant from unsloth/GLM-5.2-GGUF.
       Downloads all 11 shards and links ./ds4flash.gguf to the first shard.

  glm-antirez-iq2xxs
       GLM 5.2 antirez routed IQ2_XXS GGUF from antirez/GLM-5.2-GGUF.
       Includes Q2_K block 78 and is intended for reduced-memory testing.

  glm-antirez-q2
       GLM 5.2 antirez routed Q2_K GGUF from antirez/GLM-5.2-GGUF.
       About 262 GB on disk.

  glm-antirez-q4
       GLM 5.2 antirez routed Q4_K GGUF from antirez/GLM-5.2-GGUF.
       About 434 GB on disk.

  qwen36-q4
       Qwen3.6 27B Q4_K_M target GGUF, about 19 GB on disk.

  qwen36-q4-s
       Qwen3.6 27B Q4_K_S target GGUF from unsloth/Qwen3.6-27B-GGUF,
       about 15.9 GB on disk. Recommended for a 24 GB RTX 3090.

  qwen36-mtp
       Optional Qwen3.6 27B MTP Q4_0 GGUF, about 1.7 GB on disk. Enable it
       with --mtp; no path argument is required for the default ./gguf layout.

  qwen36-q4-mtp
       Downloads both Qwen3.6 files, verifies their SHA-256 checksums, and
       links ./ds4flash.gguf to the Q4_K_M target model.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp-model <download directory>/$MTP_FILE --mtp-draft 2

After downloading DSpark support, enable it explicitly in greedy mode:
  ./ds4 --dspark --mtp-model <download directory>/$DSPARK_SUPPORT_FILE --temp 0

PRO and GLM files are downloaded with the official Hugging Face downloader
because they are too large, sharded, or nested for the curl path used by the
smaller DeepSeek Flash GGUF files.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
MODEL_FILES=
LINK_MODEL=1
FORCE_HF_DOWNLOAD=0
FLATTEN_DOWNLOADS=0
HF_REVISION=

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    pro-q2-imatrix) MODEL_FILE=$PRO_Q2_IMATRIX_FILE ;;
    pro-q4-layers00-30) MODEL_FILE=$PRO_Q4_LAYERS00_30_FILE; LINK_MODEL=0 ;;
    pro-q4-layers31-output) MODEL_FILE=$PRO_Q4_LAYERS31_OUTPUT_FILE; LINK_MODEL=0 ;;
    pro-q4-split)
        MODEL_FILES="$PRO_Q4_LAYERS00_30_FILE $PRO_Q4_LAYERS31_OUTPUT_FILE"
        LINK_MODEL=0
        ;;
    mtp) MODEL_FILE=$MTP_FILE; LINK_MODEL=0 ;;
    dspark-support) MODEL_FILE=$DSPARK_SUPPORT_FILE; LINK_MODEL=0 ;;
    glm-unsloth-q4)
        REPO=$GLM_UNSLOTH_REPO
        MODEL_FILE=$GLM_UNSLOTH_Q4_FIRST_FILE
        MODEL_FILES=
        for part in 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011; do
            MODEL_FILES="$MODEL_FILES $GLM_UNSLOTH_Q4_REMOTE_BASE-${part}-of-00011.gguf"
        done
        FORCE_HF_DOWNLOAD=1
        FLATTEN_DOWNLOADS=1
        ;;
    glm-antirez-q2)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-iq2xxs)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_IQ2XXS_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-q4)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    qwen36-q4)
        REPO=$QWEN36_REPO
        HF_REVISION=$QWEN36_REVISION
        MODEL_FILE=$QWEN36_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    qwen36-q4-s)
        REPO=$QWEN36_UNSLOTH_REPO
        HF_REVISION=
        MODEL_FILE=$QWEN36_Q4_S_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    qwen36-mtp)
        REPO=$QWEN36_REPO
        HF_REVISION=$QWEN36_REVISION
        MODEL_FILE=$QWEN36_MTP_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
    qwen36-q4-mtp)
        REPO=$QWEN36_REPO
        HF_REVISION=$QWEN36_REVISION
        MODEL_FILE=$QWEN36_Q4_FILE
        MODEL_FILES="$QWEN36_Q4_FILE $QWEN36_MTP_FILE"
        FORCE_HF_DOWNLOAD=1
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

needs_hf_download() {
    if [ "${FORCE_HF_DOWNLOAD:-0}" -eq 1 ]; then
        return 0
    fi
    case "$1" in
        "$PRO_Q2_IMATRIX_FILE"|"$PRO_Q4_LAYERS00_30_FILE"|"$PRO_Q4_LAYERS31_OUTPUT_FILE")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

find_hf_command() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' hf
        return 0
    fi
    for dir in "$HOME"/Library/Python/*/bin "$HOME"/.local/bin; do
        if [ -x "$dir/hf" ]; then
            printf '%s\n' "$dir/hf"
            return 0
        fi
    done
    return 1
}

local_download_name() {
    if [ "${FLATTEN_DOWNLOADS:-0}" -eq 1 ]; then
        basename "$1"
    else
        printf '%s\n' "$1"
    fi
}

download_one_hf() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    hf_out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "The Hugging Face downloader cannot resume curl .part files." >&2
        echo "Move or remove that partial download before retrying this target." >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "Large GGUF downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ] && [ -n "$HF_REVISION" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --revision "$HF_REVISION" --local-dir "$OUT_DIR" --token "$TOKEN"
    elif [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    elif [ -n "$HF_REVISION" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --revision "$HF_REVISION" --local-dir "$OUT_DIR"
    else
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ "$hf_out" != "$out" ] && [ -s "$hf_out" ]; then
        mv "$hf_out" "$out"
        rmdir "$(dirname "$hf_out")" 2>/dev/null || true
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

verify_sha256() {
    file=$1
    expected=
    case "$(basename "$file")" in
        "$QWEN36_Q4_FILE") expected=$QWEN36_Q4_SHA256 ;;
        "$QWEN36_Q4_S_FILE") expected=$QWEN36_Q4_S_SHA256 ;;
        "$QWEN36_MTP_FILE") expected=$QWEN36_MTP_SHA256 ;;
        *) return 0 ;;
    esac

    path="$OUT_DIR/$(local_download_name "$file")"
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$path" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual=$(shasum -a 256 "$path" | awk '{print $1}')
    elif command -v openssl >/dev/null 2>&1; then
        actual=$(openssl dgst -sha256 "$path" | awk '{print $NF}')
    else
        echo "Cannot verify SHA-256: install sha256sum, shasum, or openssl." >&2
        exit 1
    fi
    if [ "$actual" != "$expected" ]; then
        echo "SHA-256 mismatch for $path" >&2
        echo "Expected: $expected" >&2
        echo "Actual:   $actual" >&2
        exit 1
    fi
    echo "Verified SHA-256: $path"
}

download_one() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    if needs_hf_download "$file"; then
        download_one_hf "$file"
        return
    fi

    mkdir -p "$(dirname "$out")"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

if [ -n "$MODEL_FILES" ]; then
    for file in $MODEL_FILES; do
        download_one "$file"
        verify_sha256 "$file"
    done
else
    download_one "$MODEL_FILE"
    verify_sha256 "$MODEL_FILE"
fi

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q2-q4-imatrix, and q4-imatrix."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp-model $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" = "dspark-support" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly in greedy mode:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp-model $OUT_DIR/$DSPARK_SUPPORT_FILE --temp 0"
elif [ "$MODEL" = "qwen36-mtp" ]; then
    echo
    echo "Qwen3.6 MTP downloaded to $OUT_DIR/$QWEN36_MTP_FILE."
    echo "Enable it with ./ds4 --mtp or ./ds4-server --mtp."
elif [ "$MODEL" = "pro-q4-layers00-30" ] || [ "$MODEL" = "pro-q4-layers31-output" ] || [ "$MODEL" = "pro-q4-split" ]; then
    echo
    echo "Downloaded PRO Q4 distributed split file(s). Use them with --layers,"
    echo "for example coordinator layers 0:30 and worker layers 31:output."
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
