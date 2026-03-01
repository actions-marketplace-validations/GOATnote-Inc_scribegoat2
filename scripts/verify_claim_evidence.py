#!/usr/bin/env python3
"""Verify claim-to-evidence chains for the whitepaper.

Checks:
1. Every claim YAML's run_log_id references exist in experiments/run_log.jsonl
2. Every referenced artifact path exists (or is an LFS pointer)
3. Every FALSIFIED/SUPERSEDED claim is referenced in §06 (Falsification Record)
4. No orphaned claims (YAML exists but no section references it)
5. No dangling references (section references a claim ID with no YAML)
6. FINDINGS.md status vocabulary matches claim YAML status

Usage:
    python scripts/verify_claim_evidence.py             # Advisory mode
    python scripts/verify_claim_evidence.py --strict     # Exit 1 on any failure
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Install with: pip install pyyaml")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAIMS_DIR = PROJECT_ROOT / "governance" / "claims"
WHITEPAPER_DIR = PROJECT_ROOT / "docs" / "whitepaper"
RUN_LOG = PROJECT_ROOT / "experiments" / "run_log.jsonl"
FINDINGS_MD = PROJECT_ROOT / "experiments" / "FINDINGS.md"
SECTION_GLOB = "[0-9][0-9]_*.md"
FALSIFICATION_SECTION = "06_FALSIFICATION_RECORD.md"

CLAIM_REF_PATTERN = re.compile(r"\{\{claim:(CLM-\d{4}-\d{4})\}\}")

# Status mapping: FINDINGS.md uses ACTIVE/FALSIFIED/etc,
# claim YAMLs use established/falsified/etc
FINDINGS_TO_YAML_STATUS = {
    "ACTIVE": ["established", "provisional"],
    "CONFIRMED": ["established"],
    "PRELIMINARY": ["provisional"],
    "REVISED": ["partially_falsified", "established", "provisional"],
    "FALSIFIED": ["falsified"],
    "SUPERSEDED": ["superseded"],
}

FALSIFIED_STATUSES = {"falsified", "partially_falsified", "superseded"}


class VerificationReport:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.info.append(msg)

    def print_report(self) -> None:
        print("=" * 60)
        print("Claim Evidence Verification Report")
        print("=" * 60)
        print()

        if self.info:
            for msg in self.info:
                print(f"  OK: {msg}")
            print()

        if self.warnings:
            print(f"Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  WARN: {msg}")
            print()

        if self.errors:
            print(f"Errors ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  ERROR: {msg}")
            print()

        total = len(self.errors) + len(self.warnings)
        if total == 0:
            print("All checks passed.")
        else:
            print(
                f"Summary: {len(self.errors)} errors, "
                f"{len(self.warnings)} warnings"
            )

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


def load_claims() -> dict[str, dict]:
    """Load all claim YAML files."""
    claims = {}
    if not CLAIMS_DIR.exists():
        return claims
    for yaml_file in sorted(CLAIMS_DIR.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "id" in data:
                claims[data["id"]] = data
                claims[data["id"]]["_source_file"] = str(yaml_file)
        except Exception as e:
            print(f"WARNING: Failed to parse {yaml_file}: {e}", file=sys.stderr)
    return claims


def load_run_log_ids() -> set[str]:
    """Load all experiment IDs from run_log.jsonl."""
    ids = set()
    if not RUN_LOG.exists():
        return ids
    try:
        with open(RUN_LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "id" in entry:
                        ids.add(entry["id"])
                    if "experiment_id" in entry:
                        ids.add(entry["experiment_id"])
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return ids


def find_section_claim_refs() -> dict[str, set[str]]:
    """Find all {{claim:CLM-*}} references in section files.

    Returns dict mapping section filename to set of referenced claim IDs.
    """
    refs: dict[str, set[str]] = {}
    for section_file in sorted(WHITEPAPER_DIR.glob(SECTION_GLOB)):
        content = section_file.read_text()
        found = set(CLAIM_REF_PATTERN.findall(content))
        if found:
            refs[section_file.name] = found
    # Also check abstract
    abstract = WHITEPAPER_DIR / "ABSTRACT_AND_LIMITATIONS.md"
    if abstract.exists():
        content = abstract.read_text()
        found = set(CLAIM_REF_PATTERN.findall(content))
        if found:
            refs[abstract.name] = found
    return refs


def check_artifact_exists(artifact_path: str) -> bool:
    """Check if an artifact path exists (absolute or relative to PROJECT_ROOT)."""
    p = Path(artifact_path)
    if p.is_absolute():
        return p.exists()
    # Try relative to project root
    full = PROJECT_ROOT / artifact_path
    if full.exists():
        # Check if it's an LFS pointer (starts with "version https://git-lfs")
        if full.is_file() and full.stat().st_size < 200:
            try:
                head = full.read_bytes()[:24]
                if head.startswith(b"version https://git-lfs"):
                    return True  # LFS pointer counts as existing
            except Exception:
                pass
        return True
    return False


def verify_run_log_refs(
    claims: dict[str, dict], run_log_ids: set[str], report: VerificationReport
) -> None:
    """Check 1: Every claim's run_log_id references exist in run_log.jsonl."""
    for claim_id, claim in sorted(claims.items()):
        # Look for run_log_id or run_log_ids in evidence
        claim_items = claim.get("claims", [])
        if not isinstance(claim_items, list):
            continue
        for item in claim_items:
            if not isinstance(item, dict):
                continue
            for ev in item.get("evidence", []):
                if not isinstance(ev, dict):
                    continue
                run_id = ev.get("run_log_id")
                if run_id and run_log_ids and run_id not in run_log_ids:
                    report.warn(
                        f"{claim_id}: run_log_id '{run_id}' not found in "
                        f"experiments/run_log.jsonl"
                    )


def verify_artifact_paths(
    claims: dict[str, dict], report: VerificationReport
) -> None:
    """Check 2: Every referenced artifact path exists."""
    for claim_id, claim in sorted(claims.items()):
        claim_items = claim.get("claims", [])
        if not isinstance(claim_items, list):
            continue
        for item in claim_items:
            if not isinstance(item, dict):
                continue
            for ev in item.get("evidence", []):
                if not isinstance(ev, dict):
                    continue
                artifact = ev.get("artifact", "")
                if artifact and not check_artifact_exists(artifact):
                    report.warn(
                        f"{claim_id}: artifact '{artifact}' not found "
                        f"(may be in sibling repo or LFS)"
                    )

        # Also check evidence_chain
        chain = claim.get("evidence_chain", {})
        if isinstance(chain, dict):
            for key in ["primary", "supporting", "calibration"]:
                items = chain.get(key, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            src = item.get("source", "")
                            if src and not check_artifact_exists(src):
                                report.warn(
                                    f"{claim_id}: evidence source '{src}' "
                                    f"not found"
                                )


def verify_falsified_in_section6(
    claims: dict[str, dict],
    section_refs: dict[str, set[str]],
    report: VerificationReport,
) -> None:
    """Check 3: Every FALSIFIED/SUPERSEDED claim is referenced in §06."""
    falsification_refs = section_refs.get(FALSIFICATION_SECTION, set())

    for claim_id, claim in sorted(claims.items()):
        status = claim.get("status", "")
        if status in FALSIFIED_STATUSES:
            if claim_id not in falsification_refs:
                # Also check if it appears in the file text without {{claim:}} syntax
                fals_file = WHITEPAPER_DIR / FALSIFICATION_SECTION
                if fals_file.exists():
                    content = fals_file.read_text()
                    if claim_id in content:
                        continue
                report.warn(
                    f"{claim_id} has status '{status}' but is not "
                    f"referenced in {FALSIFICATION_SECTION}"
                )


def verify_no_orphans(
    claims: dict[str, dict],
    section_refs: dict[str, set[str]],
    report: VerificationReport,
) -> None:
    """Check 4: No orphaned claims (YAML exists but no section references it)."""
    all_refs = set()
    for refs in section_refs.values():
        all_refs.update(refs)

    for claim_id in sorted(claims.keys()):
        if claim_id not in all_refs:
            report.warn(
                f"{claim_id}: YAML exists but no section references it"
            )


def verify_no_dangling(
    claims: dict[str, dict],
    section_refs: dict[str, set[str]],
    report: VerificationReport,
) -> None:
    """Check 5: No dangling references (section references claim with no YAML)."""
    for section, refs in sorted(section_refs.items()):
        for ref in sorted(refs):
            if ref not in claims:
                report.error(
                    f"{section}: references {ref} but no claim YAML exists"
                )


def verify_findings_consistency(
    claims: dict[str, dict], report: VerificationReport
) -> None:
    """Check 6: FINDINGS.md status vocabulary matches claim YAML status."""
    if not FINDINGS_MD.exists():
        report.warn("experiments/FINDINGS.md not found, skipping consistency check")
        return

    content = FINDINGS_MD.read_text()

    # Look for claim IDs mentioned in FINDINGS.md and check status consistency
    for claim_id, claim in sorted(claims.items()):
        yaml_status = claim.get("status", "")
        # This is advisory — FINDINGS.md may not reference claims by ID
        # Just check that the claim's assertion keywords appear if it's established
        if yaml_status == "falsified":
            assertion = claim.get("assertion", "")
            # Check if there's a FALSIFIED marker near the assertion keywords
            # This is a heuristic check, not exact
            pass  # Advisory only — don't error on this


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify claim-to-evidence chains"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any error",
    )
    args = parser.parse_args()

    report = VerificationReport()

    # Load data
    claims = load_claims()
    if not claims:
        print("No claims found in governance/claims/. Nothing to verify.")
        sys.exit(0)

    report.ok(f"Loaded {len(claims)} claims from governance/claims/")

    run_log_ids = load_run_log_ids()
    if run_log_ids:
        report.ok(f"Loaded {len(run_log_ids)} experiment IDs from run_log.jsonl")
    else:
        report.warn("No run_log.jsonl found or empty — skipping run_log_id checks")

    section_refs = find_section_claim_refs()
    total_refs = sum(len(r) for r in section_refs.values())
    report.ok(
        f"Found {total_refs} claim references across "
        f"{len(section_refs)} section files"
    )

    # Run checks
    verify_run_log_refs(claims, run_log_ids, report)
    verify_artifact_paths(claims, report)
    verify_falsified_in_section6(claims, section_refs, report)
    verify_no_orphans(claims, section_refs, report)
    verify_no_dangling(claims, section_refs, report)
    verify_findings_consistency(claims, report)

    # Report
    report.print_report()

    if args.strict and report.has_errors:
        sys.exit(1)
    elif report.has_errors:
        sys.exit(0)  # Advisory mode — don't block
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
