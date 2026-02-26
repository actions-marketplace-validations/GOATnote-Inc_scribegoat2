"""OpenEM corpus bridge for bloom_eval_v2 RAG-augmented evaluation.

Provides per-scenario clinical knowledge retrieval from the OpenEM
emergency medicine corpus. Uses the canonical condition map from OpenEM
(auto-generated from corpus frontmatter + overlay.yaml).

Usage:
    from evaluation.bloom_eval_v2.openem_bridge import OpenEMBridge

    bridge = OpenEMBridge("/path/to/openem/data/index")
    context = bridge.get_context_for_scenario(scenario)
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from evaluation.bloom_eval_v2.scenarios.loader import Scenario


def _load_bridge(index_dir: Path):
    """Load OpenEMBridge from the openem package.

    Falls back to sys.path manipulation for environments where the
    openem package is not installed.
    """
    try:
        from openem import OpenEMIndex
        from openem.bridge import OpenEMBridge as _Bridge
    except ImportError:
        # Fallback: add openem-corpus/python and scripts to sys.path
        corpus_root = index_dir.parent.parent
        for subdir in ("python", "scripts"):
            p = corpus_root / subdir
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from openem import OpenEMIndex
        from openem.bridge import OpenEMBridge as _Bridge

    idx = OpenEMIndex(index_dir)
    return _Bridge(idx, fallback_separator=" ")


class OpenEMBridge:
    """Bridge between bloom_eval_v2 scenarios and OpenEM corpus."""

    def __init__(self, index_dir: str | Path):
        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(f"OpenEM index not found at {index_dir}")
        self._bridge = _load_bridge(index_dir)

    @property
    def corpus_info(self) -> str:
        return self._bridge.corpus_info

    def get_context_for_scenario(
        self,
        scenario: "Scenario",
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> Optional[str]:
        """Retrieve clinical context for a bloom_eval_v2 scenario.

        Returns formatted context string, or None if no match.
        """
        return self._bridge.get_context(scenario.condition, top_k, max_chars)

    def format_system_context(
        self,
        scenario: "Scenario",
        top_k: int = 5,
        max_chars: int = 3000,
    ) -> Optional[str]:
        """Format retrieved context as a system prompt prefix."""
        return self._bridge.format_system_context(
            scenario.condition, top_k, max_chars,
        )
