# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Rewritten `docs/whitepaper/WHITEPAPER_INDEX.md` with correct §00–§10 filenames, updated cross-references, and current descriptions
- Fixed falsification count in §00 Executive Summary (2 → 7) to match §06 Falsification Record
- Fixed heading in `.claude/commands/whitepaper-refresh.md` (removed slash prefix)
- Added whitepaper paths to CLAUDE.md Key File Paths table
- Rebuilt `docs/whitepaper/WHITEPAPER_FULL.md` from updated section files

### Added
- This changelog

## [2026-03-01] — Whitepaper Refactoring

### Changed
- Refactored whitepaper from static paper to living evidence portfolio (`fd7579da`)
  - 10 old sections → 11 new sections (§00–§10) with epistemic status tracking
  - Added claim reference system (`{{claim:CLM-*}}`) linking sections to `governance/claims/*.yaml`
  - Added `scripts/build_whitepaper_md.py` (auto-generates §01 Epistemic Map, resolves claim refs)
  - Added `scripts/verify_claim_evidence.py` (strict claim-to-evidence chain verification)
  - Added `.claude/agents/whitepaper-synthesizer.md` and `.claude/commands/whitepaper-refresh.md`
- Fixed README links to new whitepaper section paths (`e7bbb22e`)
- Replaced "Phase 1/Phase 3" labels with "Initial Evaluation/Expanded Corpus" (`08036d0d`)
