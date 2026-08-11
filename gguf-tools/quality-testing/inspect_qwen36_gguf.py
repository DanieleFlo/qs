#!/usr/bin/env python3
"""Inspect pinned Qwen3.6 GGUF artifacts without loading model tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from qwen36_fixtures import FixtureError, load_json, sha256_file, write_json


GGUF_TYPES = {
    0: ("uint8", "<B"), 1: ("int8", "<b"), 2: ("uint16", "<H"),
    3: ("int16", "<h"), 4: ("uint32", "<I"), 5: ("int32", "<i"),
    6: ("float32", "<f"), 7: ("bool", "<?"), 8: ("string", None),
    9: ("array", None), 10: ("uint64", "<Q"), 11: ("int64", "<q"),
    12: ("float64", "<d"),
}

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}

SNAPSHOT_FORMAT = "ds4-qwen36-gguf-snapshot-v1"
SEMANTIC_PREFIXES = ("general.", "qwen35.", "tokenizer.ggml.")
SEMANTIC_KEYS = {
    "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token", "tokenizer.chat_template",
}


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.fp = path.open("rb")
        self.size = path.stat().st_size

    def close(self) -> None:
        self.fp.close()

    def read(self, length: int) -> bytes:
        data = self.fp.read(length)
        if len(data) != length:
            raise FixtureError(f"{self.path}: truncated GGUF header at offset {self.fp.tell()}")
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def string(self) -> str:
        (length,) = self.unpack("<Q")
        if length > self.size:
            raise FixtureError(f"{self.path}: invalid GGUF string length {length}")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FixtureError(f"{self.path}: invalid UTF-8 in GGUF header") from exc

    def value(self, value_type: int):
        if value_type not in GGUF_TYPES:
            raise FixtureError(f"{self.path}: unsupported GGUF metadata type {value_type}")
        name, fmt = GGUF_TYPES[value_type]
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type, count = self.unpack("<IQ")
            if element_type not in GGUF_TYPES or element_type == 9:
                raise FixtureError(f"{self.path}: unsupported GGUF array element type {element_type}")
            if count > 10_000_000:
                raise FixtureError(f"{self.path}: unreasonable GGUF array length {count}")
            values = [self.value(element_type) for _ in range(count)]
            return {"element_type": GGUF_TYPES[element_type][0], "count": count, "values": values}
        assert fmt is not None
        return self.unpack(fmt)[0]


def compact_metadata(key: str, value, raw_sha256: str):
    if isinstance(value, dict) and "values" in value:
        values = value["values"]
        result = {
            "element_type": value["element_type"],
            "count": value["count"],
            "encoded_sha256": raw_sha256,
        }
        if len(values) <= 64:
            result["values"] = values
        return result
    if key == "tokenizer.chat_template":
        encoded = value.encode("utf-8")
        return {"utf8_size_bytes": len(encoded), "utf8_sha256": hashlib.sha256(encoded).hexdigest()}
    return value


def inspect_gguf(path: Path, *, include_artifact_sha256: bool = True) -> dict:
    path = path.resolve()
    reader = Reader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise FixtureError(f"{path}: invalid GGUF magic")
        version, = reader.unpack("<I")
        if version != 3:
            raise FixtureError(f"{path}: unsupported GGUF version {version}")
        tensor_count, metadata_count = reader.unpack("<QQ")
        if tensor_count > 10_000_000 or metadata_count > 1_000_000:
            raise FixtureError(f"{path}: unreasonable GGUF directory counts")

        metadata: dict[str, object] = {}
        all_keys: set[str] = set()
        for _ in range(metadata_count):
            key = reader.string()
            if key in all_keys:
                raise FixtureError(f"{path}: duplicate GGUF metadata key {key}")
            all_keys.add(key)
            value_type, = reader.unpack("<I")
            value_start = reader.fp.tell()
            value = reader.value(value_type)
            value_end = reader.fp.tell()
            reader.fp.seek(value_start)
            raw_sha256 = hashlib.sha256(reader.read(value_end - value_start)).hexdigest()
            if key.startswith(SEMANTIC_PREFIXES) or key in SEMANTIC_KEYS:
                metadata[key] = compact_metadata(key, value, raw_sha256)

        tensors = []
        tensor_names: set[str] = set()
        type_counts: dict[str, int] = {}
        previous_offset = -1
        for _ in range(tensor_count):
            name = reader.string()
            if name in tensor_names:
                raise FixtureError(f"{path}: duplicate GGUF tensor name {name}")
            tensor_names.add(name)
            dimensions, = reader.unpack("<I")
            if dimensions < 1 or dimensions > 4:
                raise FixtureError(f"{path}: invalid dimension count for tensor {name}")
            shape = list(reader.unpack("<" + "Q" * dimensions))
            tensor_type, = reader.unpack("<I")
            if tensor_type not in GGML_TYPES:
                raise FixtureError(f"{path}: unsupported GGML tensor type {tensor_type} for {name}")
            offset, = reader.unpack("<Q")
            if offset < previous_offset:
                raise FixtureError(f"{path}: tensor offsets are not monotonic at {name}")
            previous_offset = offset
            type_name = GGML_TYPES[tensor_type]
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            tensors.append({"name": name, "shape": shape, "type": type_name, "offset": offset})

        header_size = reader.fp.tell()
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1:
            raise FixtureError(f"{path}: invalid general.alignment")
        data_offset = (header_size + alignment - 1) // alignment * alignment
        if data_offset >= reader.size:
            raise FixtureError(f"{path}: tensor data starts beyond end of file")
        for tensor in tensors:
            if data_offset + tensor["offset"] >= reader.size:
                raise FixtureError(f"{path}: tensor offset outside file for {tensor['name']}")

        tokenizer_metadata = {
            key: metadata[key] for key in sorted(metadata)
            if key.startswith("tokenizer.ggml.")
        }
        tokenizer_metadata_sha256 = hashlib.sha256(json.dumps(
            tokenizer_metadata, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return {
            "format": SNAPSHOT_FORMAT,
            "artifact": {
                "filename": path.name,
                "size_bytes": reader.size,
                "sha256": sha256_file(path) if include_artifact_sha256 else None,
            },
            "gguf": {
                "version": version,
                "metadata_count": metadata_count,
                "tensor_count": tensor_count,
                "header_size_bytes": header_size,
                "data_offset_bytes": data_offset,
                "tokenizer_metadata_sha256": tokenizer_metadata_sha256,
            },
            "metadata": metadata,
            "tensor_type_counts": dict(sorted(type_counts.items())),
            "tensors": tensors,
        }
    finally:
        reader.close()


def validate_against_manifest(snapshot: dict, artifact: dict) -> None:
    actual = snapshot["artifact"]
    for field in ("filename", "size_bytes", "sha256"):
        if actual[field] != artifact[field]:
            raise FixtureError(f"artifact {field} mismatch: expected {artifact[field]!r}, got {actual[field]!r}")
    metadata = snapshot["metadata"]
    expected = artifact["metadata"]
    if metadata.get("general.architecture") != expected["general.architecture"]:
        raise FixtureError("general.architecture does not match manifest")
    if metadata.get("qwen35.context_length") != expected["context_length"]:
        raise FixtureError("qwen35.context_length does not match manifest")


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ROLE=PATH")
    role, raw_path = value.split("=", 1)
    if role not in {"target", "mtp"} or not raw_path:
        raise argparse.ArgumentTypeError("artifact role must be target or mtp")
    return role, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact", required=True, action="append", type=parse_artifact)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = load_json(args.manifest.resolve())
        declared = {item["role"]: item for item in manifest.get("artifacts", [])}
        supplied = dict(args.artifact)
        if not supplied or not set(supplied).issubset(declared):
            raise FixtureError(f"artifact roles must be selected from: {', '.join(sorted(declared))}")
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for role in sorted(supplied):
            snapshot = inspect_gguf(supplied[role])
            validate_against_manifest(snapshot, declared[role])
            output = args.snapshot_dir / f"{role}.json"
            write_json(output, snapshot)
            print(f"{role}\t{output}\t{sha256_file(output)}")
        return 0
    except (FixtureError, OSError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
