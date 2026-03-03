"""
ScribeGoat2 Utilities Module
============================

Common utilities for hashing, manifest generation, reproducibility, and I/O.
"""

from .hashing import (
    ManifestData,
    generate_manifest,
    sha256_dir_tree,
    sha256_file,
    sha256_text,
)
from .jsonl import (
    append_jsonl,
    load_jsonl,
    write_jsonl,
)

__all__ = [
    "sha256_file",
    "sha256_text",
    "sha256_dir_tree",
    "generate_manifest",
    "ManifestData",
    "load_jsonl",
    "write_jsonl",
    "append_jsonl",
]
