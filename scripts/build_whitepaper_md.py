#!/usr/bin/env python3
"""Build WHITEPAPER_FULL.md by concatenating section files.

Features:
    - Concatenates _preamble.md + ABSTRACT + numbered sections
    - Resolves {{claim:CLM-2026-NNNN}} references from governance/claims/*.yaml
    - Auto-generates §01 (Epistemic Map) from claim YAMLs
    - Validates all claim references exist
    - Warns on stale evidence (>60 days since access_date)

Usage:
    python scripts/build_whitepaper_md.py          # Write WHITEPAPER_FULL.md
    python scripts/build_whitepaper_md.py --check   # Verify committed file is up to date
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Install with: pip install pyyaml")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHITEPAPER_DIR = PROJECT_ROOT / "docs" / "whitepaper"
OUTPUT_FILE = WHITEPAPER_DIR / "WHITEPAPER_FULL.md"
PREAMBLE_FILE = WHITEPAPER_DIR / "_preamble.md"
ABSTRACT_FILE = WHITEPAPER_DIR / "ABSTRACT_AND_LIMITATIONS.md"
CLAIMS_DIR = PROJECT_ROOT / "governance" / "claims"
SECTION_GLOB = "[0-9][0-9]_*.md"
EPISTEMIC_MAP_FILE = "01_EPISTEMIC_MAP.md"

CLAIM_REF_PATTERN = re.compile(r"\{\{claim:(CLM-\d{4}-\d{4})\}\}")
DATE_LINE_PATTERN = re.compile(r"\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}")
STALENESS_DAYS = 60


def normalize_for_comparison(text: str) -> str:
    """Strip volatile metadata (auto-generated dates) for deterministic comparison."""
    return DATE_LINE_PATTERN.sub("**Last Updated:** DATE", text)


# Status display configuration
STATUS_BADGES = {
    "established": "ESTABLISHED",
    "provisional": "PROVISIONAL",
    "falsified": "FALSIFIED",
    "partially_falsified": "PARTIALLY FALSIFIED",
    "superseded": "SUPERSEDED",
}

STATUS_ORDER = [
    "established",
    "provisional",
    "partially_falsified",
    "superseded",
    "falsified",
]


def load_claims() -> dict[str, dict]:
    """Load all claim YAML files from governance/claims/."""
    claims = {}
    if not CLAIMS_DIR.exists():
        return claims
    for yaml_file in sorted(CLAIMS_DIR.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "id" in data:
                claims[data["id"]] = data
                claims[data["id"]]["_source_file"] = str(yaml_file.name)
        except Exception as e:
            print(f"WARNING: Failed to parse {yaml_file.name}: {e}", file=sys.stderr)
    return claims


def get_claim_assertion(claim: dict) -> str:
    """Extract the primary assertion text from a claim."""
    assertion = claim.get("assertion", "")
    if isinstance(assertion, str):
        return assertion.strip()
    # Handle dict assertions (e.g., superseded claims with original/current)
    if isinstance(assertion, dict):
        current = assertion.get("current", "")
        if current:
            return current.strip()
        original = assertion.get("original", "")
        if original:
            return original.strip()
    return str(assertion).strip()


def get_primary_evidence_summary(claim: dict) -> str:
    """Build a short evidence summary from the claim's quantitative sub-claims."""
    parts = []
    claim_items = claim.get("claims", [])
    if isinstance(claim_items, list):
        for item in claim_items[:2]:  # First 2 sub-claims
            if isinstance(item, dict):
                sub_assertion = item.get("assertion", "")
                if sub_assertion:
                    parts.append(sub_assertion.strip())
    return "; ".join(parts) if parts else "See claim YAML for details."


def get_evidence_sources(claim: dict) -> str:
    """Extract evidence source identifiers."""
    sources = []
    chain = claim.get("evidence_chain", {})
    if isinstance(chain, dict):
        for key in ["primary", "supporting"]:
            items = chain.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        src = item.get("source", "")
                        if src:
                            sources.append(Path(src).name)
    return ", ".join(sources[:3]) if sources else "governance/claims/"


def get_latest_access_date(claim: dict) -> str | None:
    """Find the most recent access_date in claim evidence."""
    dates = []
    claim_items = claim.get("claims", [])
    if isinstance(claim_items, list):
        for item in claim_items:
            if isinstance(item, dict):
                for ev in item.get("evidence", []):
                    if isinstance(ev, dict) and "access_date" in ev:
                        dates.append(ev["access_date"])
    return max(dates) if dates else None


def render_claim_reference(claim_id: str, claims: dict) -> str:
    """Render a {{claim:CLM-2026-NNNN}} reference as an inline badge."""
    if claim_id not in claims:
        return f"**[MISSING CLAIM: {claim_id}]**"

    claim = claims[claim_id]
    status = claim.get("status", "unknown")
    badge = STATUS_BADGES.get(status, status.upper())
    assertion = get_claim_assertion(claim)
    evidence_summary = get_primary_evidence_summary(claim)
    sources = get_evidence_sources(claim)
    access_date = get_latest_access_date(claim)
    date_str = f" | Last verified: {access_date}" if access_date else ""

    return f"> **[{badge}]** {assertion}\n> {evidence_summary}\n> *Evidence: {sources}{date_str}*"


def resolve_claim_references(text: str, claims: dict) -> tuple[str, set[str]]:
    """Replace all {{claim:CLM-*}} references with rendered badges.

    Returns (resolved_text, set_of_referenced_claim_ids).
    """
    referenced = set()

    def replacer(match: re.Match) -> str:
        claim_id = match.group(1)
        referenced.add(claim_id)
        return render_claim_reference(claim_id, claims)

    resolved = CLAIM_REF_PATTERN.sub(replacer, text)
    return resolved, referenced


def generate_epistemic_map(claims: dict) -> str:
    """Auto-generate the epistemic map content from claim YAMLs."""
    lines = [
        "# What We Know, Believe, and Don't Know",
        "",
        "**Document Status:** AUTO-GENERATED from governance/claims/*.yaml  ",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "> This section is auto-generated by `scripts/build_whitepaper_md.py`.",
        "> To update, modify the claim YAML files in `governance/claims/` and rebuild.",
        "",
    ]

    # Group claims by status
    grouped: dict[str, list[dict]] = {}
    for claim_id, claim in sorted(claims.items()):
        status = claim.get("status", "unknown")
        grouped.setdefault(status, []).append(claim)

    # Render in status order
    status_descriptions = {
        "established": "Replicated across multiple experiments, models, or corpora. High confidence.",
        "provisional": "Single experiment or limited replication. Directionally supported but needs confirmation.",
        "partially_falsified": "Some aspects confirmed, others refuted. Nuanced status.",
        "superseded": "Originally established but replaced by stronger evidence or larger-scale findings.",
        "falsified": "Tested and found wrong. Mechanism of error documented in §06.",
    }

    status_labels = {
        "established": "ESTABLISHED",
        "provisional": "PROVISIONAL",
        "partially_falsified": "PARTIALLY FALSIFIED",
        "superseded": "SUPERSEDED",
        "falsified": "FALSIFIED",
    }

    for status in STATUS_ORDER:
        if status not in grouped:
            continue
        label = status_labels.get(status, status.upper())
        desc = status_descriptions.get(status, "")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"*{desc}*")
        lines.append("")

        for claim in grouped[status]:
            claim_id = claim.get("id", "?")
            assertion = get_claim_assertion(claim)
            # Truncate long assertions
            if len(assertion) > 200:
                assertion = assertion[:197] + "..."
            source_file = claim.get("_source_file", "")
            lines.append(f"- **{claim_id}:** {assertion}")
            if source_file:
                lines.append(f"  *(Source: `governance/claims/{source_file}`)*")
            lines.append("")

    # Summary stats
    total = len(claims)
    counts = {s: len(grouped.get(s, [])) for s in STATUS_ORDER}
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in STATUS_ORDER:
        if counts.get(status, 0) > 0:
            lines.append(f"| {status_labels.get(status, status)} | {counts[status]} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    return "\n".join(lines)


def check_staleness(claims: dict) -> list[str]:
    """Check for claims with evidence older than STALENESS_DAYS."""
    warnings = []
    cutoff = datetime.now() - timedelta(days=STALENESS_DAYS)

    for claim_id, claim in sorted(claims.items()):
        access_date = get_latest_access_date(claim)
        if access_date:
            try:
                dt = datetime.strptime(access_date, "%Y-%m-%d")
                if dt < cutoff:
                    days_old = (datetime.now() - dt).days
                    warnings.append(
                        f"  STALE: {claim_id} — last evidence access {access_date} "
                        f"({days_old} days ago, threshold {STALENESS_DAYS})"
                    )
            except ValueError:
                pass
    return warnings


def strip_metadata(text: str) -> str:
    """Remove the metadata block between the title line and the first '---' separator."""
    lines = text.split("\n")
    title_line = lines[0]
    sep_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            sep_idx = i
            break
    if sep_idx is None:
        return text
    rest = "\n".join(lines[sep_idx + 1 :]).lstrip("\n")
    return title_line + "\n\n" + rest


def strip_next_section(text: str) -> str:
    """Remove the '**Next Section:**' navigation link at the bottom."""
    return re.sub(r"\n\*\*Next Section:\*\*[^\n]*\n?$", "\n", text)


def build() -> tuple[str, list[str]]:
    """Build the full whitepaper content from section files.

    Returns (content, warnings).
    """
    warnings: list[str] = []
    claims = load_claims()
    all_referenced: set[str] = set()

    parts: list[str] = []

    # 1. Preamble
    parts.append(PREAMBLE_FILE.read_text().rstrip())

    # 2. Abstract
    abstract = ABSTRACT_FILE.read_text()
    abstract = strip_metadata(abstract)
    resolved, refs = resolve_claim_references(abstract, claims)
    all_referenced.update(refs)
    parts.append(resolved.rstrip())

    # 3. Numbered sections (sorted lexicographically)
    section_files = sorted(WHITEPAPER_DIR.glob(SECTION_GLOB))
    for section_file in section_files:
        if section_file.name == EPISTEMIC_MAP_FILE:
            # Auto-generate §01 from claim YAMLs
            content = generate_epistemic_map(claims)
        else:
            content = section_file.read_text()
            content = strip_metadata(content)
            content = strip_next_section(content)

        # Resolve claim references in all sections
        resolved, refs = resolve_claim_references(content, claims)
        all_referenced.update(refs)
        parts.append(resolved.rstrip())

    # 4. Validate claim references
    missing = all_referenced - set(claims.keys())
    if missing:
        for m in sorted(missing):
            warnings.append(f"  MISSING CLAIM: {m} referenced in section but no YAML exists")

    # 5. Check for orphaned claims (YAML exists but never referenced)
    orphaned = set(claims.keys()) - all_referenced
    if orphaned:
        for o in sorted(orphaned):
            warnings.append(f"  ORPHAN: {o} has YAML but is not referenced in any section")

    # 6. Check staleness
    stale = check_staleness(claims)
    warnings.extend(stale)

    return "\n\n---\n\n".join(parts) + "\n", warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WHITEPAPER_FULL.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that committed WHITEPAPER_FULL.md matches generated output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing claim references as errors (exit 1)",
    )
    args = parser.parse_args()

    generated, warnings = build()

    # Print warnings
    has_errors = False
    if warnings:
        print("Build warnings:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)
            if "MISSING CLAIM" in w:
                has_errors = True
        print(file=sys.stderr)

    if args.strict and has_errors:
        print(
            "ERROR: Missing claim references found. Fix before publishing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.check:
        if not OUTPUT_FILE.exists():
            print(f"ERROR: {OUTPUT_FILE} does not exist. Run without --check to generate it.")
            sys.exit(1)
        committed = OUTPUT_FILE.read_text()
        if normalize_for_comparison(committed) == normalize_for_comparison(generated):
            print("WHITEPAPER_FULL.md is up to date.")
            sys.exit(0)
        else:
            print(
                "ERROR: WHITEPAPER_FULL.md is stale. "
                "Run 'python scripts/build_whitepaper_md.py' to regenerate."
            )
            sys.exit(1)
    else:
        OUTPUT_FILE.write_text(generated)
        n_claims = len(load_claims())
        print(f"Wrote {OUTPUT_FILE}")
        print(f"  {n_claims} claims loaded from governance/claims/")
        if warnings:
            print(f"  {len(warnings)} warnings (see stderr)")


if __name__ == "__main__":
    main()
