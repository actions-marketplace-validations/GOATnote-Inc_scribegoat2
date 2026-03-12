#!/usr/bin/env python3
"""Generate a formatted physician worksheet for Phase 1 rubric decisions.

Loads RD YAMLs (sorted by phase_order), pairs each with its exemplar case
transcript from the session JSON, and prints a structured presentation for
physician review. Optionally writes to a markdown worksheet file.

Usage:
    python scripts/present_rubric_decisions.py
    python scripts/present_rubric_decisions.py --session path/to/session.json
    python scripts/present_rubric_decisions.py --output worksheet.md
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bloom_eval_v2.calibration.adjudication_system import (
    AdjudicationSession,
)

# =============================================================================
# CONSTANTS
# =============================================================================

CALIBRATION_DIR = PROJECT_ROOT / "evaluation" / "bloom_eval_v2" / "calibration"
RUBRIC_DECISIONS_DIR = CALIBRATION_DIR / "rubric_decisions"
DEFAULT_SESSION_PATH = CALIBRATION_DIR / "adjudication_session_2.json"

# Exemplar case ID -> RD mapping (from prepare_session2.py lines 57-72)
EXEMPLAR_RD_MAP = {
    "claude_MTR-001_t4": "RD-003",
    "claude_MTR-001_t3": "RD-002",
    "claude_MTR-002_t1": "RD-006",
    "claude_MTR-008_t1": "RD-004",
    "claude_MTR-023_t1": "RD-001",
}

# Reverse: RD -> exemplar case ID
RD_EXEMPLAR_MAP = {v: k for k, v in EXEMPLAR_RD_MAP.items()}


# =============================================================================
# YAML PARSING (no external dependency)
# =============================================================================


def parse_rd_yaml(path):
    """Parse a rubric decision YAML file into a dict.

    Handles multi-line scalars using '>' (folded block scalar) and
    block-style lists (key followed by indented '- item' lines).
    """
    rd = {"_path": str(path)}
    content = path.read_text()
    current_key = None
    current_mode = None  # "scalar" or "list"
    current_value_lines = []

    def _flush():
        if current_key is not None:
            if current_mode == "scalar" and current_value_lines:
                rd[current_key] = " ".join(current_value_lines).strip()
            # list mode items are appended directly to rd[current_key]

    for line in content.split("\n"):
        stripped = line.strip()
        # Skip blank lines inside multi-line values
        if not stripped:
            if current_key and current_mode == "scalar" and current_value_lines:
                _flush()
                current_key = None
                current_mode = None
                current_value_lines = []
            continue

        # Check if this is a top-level key (not indented)
        if not line.startswith(" ") and not line.startswith("\t") and ":" in stripped:
            _flush()
            current_key = None
            current_mode = None
            current_value_lines = []

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            if value == ">" or value == "|":
                # Multi-line scalar follows
                current_key = key
                current_mode = "scalar"
            elif value.startswith("["):
                # Inline list
                rd[key] = [
                    item.strip().strip("'\"")
                    for item in value.strip("[]").split(",")
                    if item.strip()
                ]
            elif value == "":
                # Empty value — could be a block list or null
                # Prepare for possible list items
                current_key = key
                current_mode = "list"
                rd[key] = []
            else:
                # Simple scalar
                value = value.strip("'\"")
                if value == "null":
                    rd[key] = None
                else:
                    rd[key] = value
        elif stripped.startswith("- "):
            # List item
            item = stripped[2:].strip().strip("'\"")
            if current_key is not None and current_mode == "list":
                rd[current_key].append(item)
            else:
                # Fallback: find the most recent list key
                for k in reversed(list(rd.keys())):
                    if isinstance(rd.get(k), list):
                        rd[k].append(item)
                        break
        else:
            # Continuation of multi-line scalar
            if current_key is not None and current_mode == "scalar":
                current_value_lines.append(stripped)

    _flush()

    # Clean up empty lists that were actually null
    for key in list(rd.keys()):
        if isinstance(rd[key], list) and len(rd[key]) == 0 and key != "_path":
            rd[key] = None

    return rd


def load_phase1_decisions(rd_dir=None):
    """Load Phase 1 rubric decisions, sorted by phase_order."""
    rd_dir = rd_dir or RUBRIC_DECISIONS_DIR
    decisions = []
    if not rd_dir.exists():
        return decisions

    for rd_path in sorted(rd_dir.glob("RD-*.yaml")):
        rd = parse_rd_yaml(rd_path)
        # Only include Phase 1 decisions (session 2, phase 1)
        if rd.get("session") == "2" and rd.get("phase") == "1":
            decisions.append(rd)

    # Sort by phase_order
    decisions.sort(key=lambda d: int(d.get("phase_order", 999)))
    return decisions


# =============================================================================
# TRANSCRIPT FORMATTING
# =============================================================================


def format_transcript(case, max_turns=None):
    """Format a case's turns into readable transcript text."""
    if not case or not case.turns:
        return "(No transcript available)"

    lines = []
    turns = case.turns if max_turns is None else case.turns[:max_turns]
    for turn in turns:
        header = f"Turn {turn.turn_number}"
        if turn.classification:
            header += f" [{turn.classification}]"
        if turn.confidence is not None:
            header += f" (conf: {turn.confidence:.2f})"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(turn.response_text.strip())
        if turn.reasoning:
            lines.append(f"  Judge reasoning: {turn.reasoning.strip()}")
        lines.append("")
    return "\n".join(lines)


def format_transcript_md(case, max_turns=None):
    """Format a case's turns into markdown transcript."""
    if not case or not case.turns:
        return "> *(No transcript available)*\n"

    lines = []
    turns = case.turns if max_turns is None else case.turns[:max_turns]
    for turn in turns:
        header = f"**Turn {turn.turn_number}**"
        if turn.classification:
            header += f" `[{turn.classification}]`"
        if turn.confidence is not None:
            header += f" *(conf: {turn.confidence:.2f})*"
        lines.append(header)
        lines.append("")
        # Blockquote the response text
        for text_line in turn.response_text.strip().split("\n"):
            lines.append(f"> {text_line}")
        if turn.reasoning:
            lines.append(f"> *Judge reasoning: {turn.reasoning.strip()}*")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# PRESENTATION (STDOUT)
# =============================================================================


def present_to_stdout(decisions, session_cases):
    """Print the structured Phase 1 rubric decision worksheet to stdout."""
    print("=" * 72)
    print("PHASE 1: RUBRIC DECISION WORKSHEET")
    print("=" * 72)
    print()
    print(f"Decisions to resolve: {len(decisions)}")
    print("Instructions: For each decision, review the exemplar transcript,")
    print("then record your ruling and reasoning.")
    print()

    for i, rd in enumerate(decisions, 1):
        rd_id = rd.get("decision_id", "?")
        question = rd.get("question", "(no question)")
        clinical_context = rd.get("clinical_context", "(no context)")
        downstream_impact = rd.get("downstream_impact", "(no impact assessment)")
        affected = rd.get("affected_scenarios", [])
        exemplar_case_id = RD_EXEMPLAR_MAP.get(rd_id)

        print(f"{'─' * 72}")
        print(f"  {rd_id} (Order: {i}/{len(decisions)})")
        print(f"{'─' * 72}")
        print()

        # Question
        print("  QUESTION:")
        _print_wrapped(question, indent=4)
        print()

        # Exemplar transcript
        print(f"  EXEMPLAR CASE: {exemplar_case_id or '(none)'}")
        if exemplar_case_id and exemplar_case_id in session_cases:
            case = session_cases[exemplar_case_id]
            print(f"  Scenario: {case.scenario_id} — {case.condition}")
            print(f"  Model: {case.model_id}")
            print(f"  Grader outcome: {case.grader_outcome}")
            print()
            transcript = format_transcript(case)
            for line in transcript.split("\n"):
                print(f"    {line}")
        elif exemplar_case_id:
            print("  (Exemplar case not found in session)")
        print()

        # Clinical context
        print("  CLINICAL CONTEXT:")
        _print_wrapped(clinical_context, indent=4)
        print()

        # Downstream impact
        print("  DOWNSTREAM IMPACT:")
        _print_wrapped(downstream_impact, indent=4)
        if affected:
            print(f"    Affected scenarios: {', '.join(affected)}")
        print()

        # Ruling blanks
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │ RULING:                                                 │")
        print("  │                                                         │")
        print("  │ REASONING:                                              │")
        print("  │                                                         │")
        print("  │                                                         │")
        print("  │ CONFIDENCE: ___  (high / medium / low)                  │")
        print("  └─────────────────────────────────────────────────────────┘")
        print()

    print("=" * 72)
    print(f"End of Phase 1 worksheet ({len(decisions)} decisions)")
    print("=" * 72)


def _print_wrapped(text, indent=4, width=68):
    """Print text with wrapping and indentation."""
    prefix = " " * indent
    words = text.split()
    line = prefix
    for word in words:
        if len(line) + len(word) + 1 > width + indent:
            print(line)
            line = prefix + word
        else:
            if line == prefix:
                line += word
            else:
                line += " " + word
    if line.strip():
        print(line)


# =============================================================================
# PRESENTATION (MARKDOWN)
# =============================================================================


def generate_worksheet_md(decisions, session_cases):
    """Generate a markdown worksheet for Phase 1 rubric decisions."""
    lines = []
    lines.append("# Phase 1: Rubric Decision Worksheet")
    lines.append("")
    lines.append(f"**Decisions to resolve:** {len(decisions)}")
    lines.append("")
    lines.append(
        "Instructions: For each decision, review the exemplar transcript, "
        "then record your ruling and reasoning."
    )
    lines.append("")

    for i, rd in enumerate(decisions, 1):
        rd_id = rd.get("decision_id", "?")
        question = rd.get("question", "(no question)")
        clinical_context = rd.get("clinical_context", "(no context)")
        downstream_impact = rd.get("downstream_impact", "(no impact assessment)")
        affected = rd.get("affected_scenarios", [])
        exemplar_case_id = RD_EXEMPLAR_MAP.get(rd_id)

        lines.append("---")
        lines.append("")
        lines.append(f"## {rd_id} (Order: {i}/{len(decisions)})")
        lines.append("")

        # Question
        lines.append("### Question")
        lines.append("")
        lines.append(question)
        lines.append("")

        # Exemplar transcript
        lines.append("### Exemplar Transcript")
        lines.append("")
        if exemplar_case_id and exemplar_case_id in session_cases:
            case = session_cases[exemplar_case_id]
            lines.append(f"**Case:** `{exemplar_case_id}`")
            lines.append(f"**Scenario:** {case.scenario_id} — {case.condition}")
            lines.append(f"**Model:** {case.model_id}")
            lines.append(f"**Grader outcome:** `{case.grader_outcome}`")
            lines.append("")
            lines.append(format_transcript_md(case))
        elif exemplar_case_id:
            lines.append(f"**Case:** `{exemplar_case_id}` *(not found in session)*")
        else:
            lines.append("*(No exemplar case assigned)*")
        lines.append("")

        # Clinical context
        lines.append("### Clinical Context")
        lines.append("")
        lines.append(clinical_context)
        lines.append("")

        # Downstream impact
        lines.append("### Downstream Impact")
        lines.append("")
        lines.append(downstream_impact)
        if affected:
            lines.append("")
            lines.append(f"**Affected scenarios:** {', '.join(affected)}")
        lines.append("")

        # Ruling blanks
        lines.append("### Ruling")
        lines.append("")
        lines.append("**Decision:** *(capitulation / harm reduction / context-dependent / ...)*")
        lines.append("")
        lines.append("**Reasoning:**")
        lines.append("")
        lines.append("")
        lines.append("**Confidence:** *(high / medium / low)*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*End of Phase 1 worksheet ({len(decisions)} decisions)*")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 rubric decision worksheet for physician review."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help=f"Path to session JSON (default: {DEFAULT_SESSION_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown worksheet to file (default: stdout only)",
    )
    parser.add_argument(
        "--rd-dir",
        type=Path,
        default=None,
        help=f"Rubric decisions directory (default: {RUBRIC_DECISIONS_DIR.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    rd_dir = args.rd_dir or RUBRIC_DECISIONS_DIR
    if not rd_dir.exists():
        print(f"ERROR: Rubric decisions directory not found: {rd_dir}", file=sys.stderr)
        sys.exit(1)

    # Load Phase 1 decisions
    decisions = load_phase1_decisions(rd_dir)
    if not decisions:
        print("No Phase 1 rubric decisions found.", file=sys.stderr)
        sys.exit(1)

    # Load session for exemplar transcripts
    session_path = args.session or DEFAULT_SESSION_PATH
    session_cases = {}
    if session_path.exists():
        session = AdjudicationSession.load(session_path)
        session_cases = {c.case_id: c for c in session.cases}
        print(f"Loaded session: {session.session_id} ({len(session.cases)} cases)", file=sys.stderr)
    else:
        # Try the source session (Session 1) as fallback
        source_path = CALIBRATION_DIR / "adjudication_session_opus46_feb17_regraded.json"
        if source_path.exists():
            session = AdjudicationSession.load(source_path)
            session_cases = {c.case_id: c for c in session.cases}
            print(
                f"Loaded source session: {session.session_id} ({len(session.cases)} cases)",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: No session file found — transcripts will be unavailable", file=sys.stderr
            )

    # Present to stdout
    present_to_stdout(decisions, session_cases)

    # Optionally write markdown
    if args.output:
        md_content = generate_worksheet_md(decisions, session_cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md_content)
        print(f"\nWorksheet written to: {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
