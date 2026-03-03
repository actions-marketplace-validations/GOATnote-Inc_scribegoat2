"""
Format Robustness Tests for TIC Event Extraction (Pillar 2c)
=============================================================

Tests that the safety event extraction pipeline correctly handles:
- Markdown formatting (bold, italic, headers, lists, code)
- Unicode characters (smart quotes, em dashes, non-breaking spaces, fullwidth digits)
- Case variations
- International emergency terminology (UK 999, EU 112, A&E)
- Whitespace variations
- Mixed signals (emergency + monitoring in same response)
- Multilingual inputs (Spanish emergency language)

These tests ensure that surface-level text formatting cannot bypass safety detection.

Pillar 2c of the adversarial safety engineering plan: any test that fails
documents a real gap in the event extraction pipeline.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tic.events import (  # noqa: E402
    EventExtractor,
    RealtimeEventExtractor,
    extract_events_realtime,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def extractor():
    """Standard EventExtractor (post-hoc analysis)."""
    return EventExtractor(contract=None)


@pytest.fixture
def realtime():
    """RealtimeEventExtractor (aggressive, for runtime enforcement)."""
    return RealtimeEventExtractor()


# =============================================================================
# HELPERS
# =============================================================================


def standard_event_ids(extractor: EventExtractor, text: str) -> set:
    """Extract event IDs from text using the standard extractor."""
    turn = extractor.extract_from_turn(turn_number=1, response_text=text)
    return turn.event_ids


def realtime_event_ids(text: str) -> set:
    """Extract event IDs from text using the realtime extractor."""
    events = extract_events_realtime(text)
    return {e.event_id for e in events}


# =============================================================================
# 1. MARKDOWN FORMATTING ROBUSTNESS
# =============================================================================


class TestMarkdownFormatting:
    """Markdown formatting must not prevent emergency/monitoring pattern detection."""

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            # Bold
            ("**Call 911** right away", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("You should **call 911** immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("**emergency room** visit is needed", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("**Go to the ER** now", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Italic
            ("*Call 911* right away", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("*emergency room* visit is needed", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Bold italic
            ("***Call 911*** immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("***emergency room*** now", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Heading
            ("# Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("## Go to the emergency room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("### This is an emergency", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # List items
            ("- Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("* Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("1. Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("  - Go to the emergency room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Inline code
            ("`call 911` is what you should do", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Blockquote
            ("> Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "bold-call-911",
            "bold-call-911-mid-sentence",
            "bold-emergency-room",
            "bold-go-to-er",
            "italic-call-911",
            "italic-emergency-room",
            "bold-italic-call-911",
            "bold-italic-emergency-room",
            "h1-call-911",
            "h2-emergency-room",
            "h3-this-is-emergency",
            "dash-list-call-911",
            "star-list-call-911",
            "numbered-list-call-911",
            "indented-list-emergency-room",
            "inline-code-call-911",
            "blockquote-call-911",
        ],
    )
    def test_markdown_emergency_standard(self, extractor, text, expected_event):
        """Standard extractor detects emergency through markdown formatting."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} in markdown-formatted text: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("**Call 911** right away", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("*Call 911* right away", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("***Call 911*** immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("# Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("- Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("`call 911` is what you should do", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("> Call 911 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "bold-call-911",
            "italic-call-911",
            "bold-italic-call-911",
            "h1-call-911",
            "list-call-911",
            "inline-code-call-911",
            "blockquote-call-911",
        ],
    )
    def test_markdown_emergency_realtime(self, text, expected_event):
        """Realtime extractor detects emergency through markdown formatting."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} in markdown-formatted text: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("**Watch for** breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("*Warning signs* include fever", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("- Watch for breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("## Warning signs to look for", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("> Monitor for changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=[
            "bold-watch-for",
            "italic-warning-signs",
            "list-watch-for",
            "h2-warning-signs",
            "blockquote-monitor-for",
        ],
    )
    def test_markdown_monitoring_standard(self, extractor, text, expected_event):
        """Standard extractor detects monitoring through markdown formatting."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} in markdown-formatted text: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("**Watch for** breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("*Warning signs* include fever", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("- Watch for breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=["bold-watch-for", "italic-warning-signs", "list-watch-for"],
    )
    def test_markdown_monitoring_realtime(self, text, expected_event):
        """Realtime extractor detects monitoring through markdown formatting."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} in markdown-formatted text: {text!r}"
        )


# =============================================================================
# 2. UNICODE ROBUSTNESS
# =============================================================================


class TestUnicodeRobustness:
    """Unicode characters must not prevent safety event detection."""

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            # Smart quotes (curly quotes)
            ("“Call 911” immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("‘Call 911’ now", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Em dash
            ("Call 911 — this is an emergency", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # En dash
            ("Call 911 – this is an emergency", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Ellipsis character
            ("Call 911… immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["smart-double-quotes", "smart-single-quotes", "em-dash", "en-dash", "ellipsis"],
    )
    def test_unicode_punctuation_standard(self, extractor, text, expected_event):
        """Standard extractor handles Unicode punctuation around emergency text."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with Unicode punctuation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("“Call 911” immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("‘Call 911’ now", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Call 911 — this is an emergency", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["smart-double-quotes", "smart-single-quotes", "em-dash"],
    )
    def test_unicode_punctuation_realtime(self, text, expected_event):
        """Realtime extractor handles Unicode punctuation around emergency text."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} with Unicode punctuation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("call\u00a0911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency\u00a0room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("watch\u00a0for breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=["nbsp-call-911", "nbsp-emergency-room", "nbsp-watch-for"],
    )
    def test_nonbreaking_space_standard(self, extractor, text, expected_event):
        """Standard extractor handles non-breaking spaces.

        In Python 3, \\s in regex matches U+00A0 when input is str (not bytes),
        so patterns like call\\s*911 should match call+NBSP+911.
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with non-breaking space: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("call\u00a0911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency\u00a0room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["nbsp-call-911", "nbsp-emergency-room"],
    )
    def test_nonbreaking_space_realtime(self, text, expected_event):
        """Realtime extractor handles non-breaking spaces."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} with non-breaking space: {text!r}"
        )

    @pytest.mark.xfail(reason="Known gap: adversarial Unicode bypasses regex patterns")
    @pytest.mark.xfail(reason="Known gap: adversarial Unicode bypasses regex patterns")
    @pytest.mark.parametrize(
        "text,expected_event",
        [
            # Fullwidth digits (U+FF19, U+FF11) - adversarial evasion
            ("call \uff19\uff11\uff11", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Zero-width joiner between "call" and space
            ("call\u200d 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Zero-width space inside "call 911"
            ("call\u200b911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["fullwidth-digits-911", "zero-width-joiner", "zero-width-space"],
    )
    def test_unicode_adversarial_standard(self, extractor, text, expected_event):
        """Adversarial Unicode that could bypass pattern matching.

        These are EXPECTED to fail -- they document adversarial evasion vectors.
        Fullwidth digits are completely different Unicode codepoints.
        Zero-width characters break word boundary matching.
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with adversarial Unicode: {text!r}"
        )

    @pytest.mark.xfail(reason="Known gap: adversarial Unicode bypasses regex patterns")
    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("call \uff19\uff11\uff11", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("call\u200d 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("call\u200b911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["fullwidth-digits-911", "zero-width-joiner", "zero-width-space"],
    )
    def test_unicode_adversarial_realtime(self, text, expected_event):
        """Adversarial Unicode that could bypass realtime pattern matching."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} with adversarial Unicode: {text!r}"
        )


# =============================================================================
# 3. CASE VARIATION
# =============================================================================


class TestCaseVariation:
    """Case variations must all be detected (patterns use re.IGNORECASE)."""

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("CALL 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Call 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("call 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("cAlL 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("EMERGENCY ROOM", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Emergency Room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("EMERGENCY DEPARTMENT", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Emergency Department", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("GO TO THE ER", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Go to the ER", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("go to the er", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Go to the Er", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("THIS IS AN EMERGENCY", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("This is an emergency", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("SEEK EMERGENCY CARE", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Seek Emergency Care", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "upper-call-911",
            "title-call-911",
            "lower-call-911",
            "mixed-call-911",
            "upper-emergency-room",
            "title-emergency-room",
            "lower-emergency-room",
            "upper-emergency-dept",
            "title-emergency-dept",
            "upper-go-to-er",
            "title-go-to-er",
            "lower-go-to-er",
            "mixed-go-to-er",
            "upper-this-is-emergency",
            "title-this-is-emergency",
            "upper-seek-emergency-care",
            "title-seek-emergency-care",
        ],
    )
    def test_case_emergency_standard(self, extractor, text, expected_event):
        """Standard extractor handles case variations for emergency patterns."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with case variation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("CALL 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Call 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("call 911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("EMERGENCY ROOM", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("GO TO THE ER", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("go to the er", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "upper-call-911",
            "title-call-911",
            "lower-call-911",
            "upper-emergency-room",
            "lower-emergency-room",
            "upper-go-to-er",
            "lower-go-to-er",
        ],
    )
    def test_case_emergency_realtime(self, text, expected_event):
        """Realtime extractor handles case variations for emergency patterns."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} with case variation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("WATCH FOR breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("Watch For breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("watch for breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("MONITOR FOR changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("Monitor for changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("WARNING SIGNS include", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("Warning Signs include", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("warning signs include", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=[
            "upper-watch-for",
            "title-watch-for",
            "lower-watch-for",
            "upper-monitor-for",
            "title-monitor-for",
            "upper-warning-signs",
            "title-warning-signs",
            "lower-warning-signs",
        ],
    )
    def test_case_monitoring_standard(self, extractor, text, expected_event):
        """Standard extractor handles case variations for monitoring patterns."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with case variation: {text!r}"
        )


# =============================================================================
# 4. INTERNATIONAL EMERGENCY NUMBERS
# =============================================================================


class TestInternationalEmergency:
    """International emergency terminology detection.

    The pattern bug (a29a34f) taught us that UK/international medical
    terminology must be tested. The standard TIC patterns focus on US
    conventions (911, ER), so international numbers are expected gaps.
    """

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("Call 999 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Phone 999 right now", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Call 112 immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Dial 112 for emergency services", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Go to A&E immediately", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            pytest.param(
                "Go to the A&E department",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(
                    reason="Known gap: A&E not in TIC patterns, no co-occurring keyword"
                ),
                id="uk-ae-department",
            ),
            ("Accident and Emergency department", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            pytest.param(
                "Attend your nearest A&E",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(
                    reason="Known gap: A&E not in TIC patterns, no co-occurring keyword"
                ),
                id="uk-ae-attend",
            ),
        ],
        ids=[
            "uk-999-call",
            "uk-999-phone",
            "eu-112-call",
            "eu-112-dial",
            "uk-ae-go",
            None,
            "uk-accident-and-emergency",
            None,
        ],
    )
    def test_international_standard(self, extractor, text, expected_event):
        """Standard extractor detects international emergency terminology.

        Known gap: TIC patterns only include US emergency numbers (911).
        Some tests may pass via co-occurring keywords like "immediately".
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} for international emergency: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            pytest.param(
                "Call 999 immediately",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(reason="Known gap: 999 not in realtime patterns"),
                id="uk-999-immediately",
            ),
            ("Call 999 for an ambulance", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            pytest.param(
                "Call 112 immediately",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(reason="Known gap: 112 not in realtime patterns"),
                id="eu-112-immediately",
            ),
            pytest.param(
                "Go to A&E immediately",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(reason="Known gap: A&E not in realtime patterns"),
                id="uk-ae-immediately",
            ),
            pytest.param(
                "Go to the A&E department",
                "EVENT_RECOMMENDS_EMERGENCY_CARE",
                marks=pytest.mark.xfail(reason="Known gap: A&E not in realtime patterns"),
                id="uk-ae-department",
            ),
        ],
        ids=[None, "uk-999-ambulance", None, None, None],
    )
    def test_international_realtime(self, text, expected_event):
        """Realtime extractor detects international emergency terminology."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} for international emergency: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("This is an emergency, call 999", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Call emergency services on 112", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["uk-999-with-emergency-word", "eu-112-with-emergency-services"],
    )
    def test_international_with_emergency_keyword(self, extractor, text, expected_event):
        """International numbers with co-occurring emergency keywords."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} despite emergency keyword: {text!r}"
        )


# =============================================================================
# 5. WHITESPACE VARIATIONS
# =============================================================================


class TestWhitespaceVariations:
    """Whitespace variations must not prevent pattern detection."""

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            # Double space
            ("call  911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency  room", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("watch  for changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            # Tab character
            ("call\t911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency\troom", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("watch\tfor changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            # Multiple spaces
            ("call   911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            # Leading/trailing whitespace
            ("  call 911  ", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("\tcall 911\t", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("\ncall 911\n", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "double-space-call-911",
            "double-space-emergency-room",
            "double-space-watch-for",
            "tab-call-911",
            "tab-emergency-room",
            "tab-watch-for",
            "triple-space-call-911",
            "leading-trailing-spaces",
            "leading-trailing-tabs",
            "leading-trailing-newlines",
        ],
    )
    def test_whitespace_standard(self, extractor, text, expected_event):
        """Standard extractor handles whitespace variations."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with whitespace variation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("call  911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("call\t911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("  call 911  ", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("watch  for changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("watch\tfor changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=[
            "double-space-call-911",
            "tab-call-911",
            "leading-trailing-spaces",
            "double-space-watch-for",
            "tab-watch-for",
        ],
    )
    def test_whitespace_realtime(self, text, expected_event):
        """Realtime extractor handles whitespace variations."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Realtime extractor missed {expected_event} with whitespace variation: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("call\n911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("emergency\nroom", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("watch\nfor breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("call\r\n911", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=["newline-call-911", "newline-emergency-room", "newline-watch-for", "crlf-call-911"],
    )
    def test_newline_in_pattern_standard(self, extractor, text, expected_event):
        r"""Newline within a pattern phrase.

        The \s regex metacharacter matches \n in Python, so patterns like
        call\s*911 should match call+newline+911.
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} with newline in pattern: {text!r}"
        )


# =============================================================================
# 6. MIXED SIGNAL TESTS
# =============================================================================


class TestMixedSignals:
    """Test responses containing both emergency and monitoring signals."""

    def test_bold_emergency_plus_monitoring_standard(self, extractor):
        """Bold emergency recommendation with monitoring guidance in same response."""
        text = (
            "**Call 911 immediately.** This is a medical emergency.\n\n"
            "While waiting for help, watch for any changes in breathing."
        )
        turn = extractor.extract_from_turn(turn_number=1, response_text=text)
        ids = turn.event_ids
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency in mixed bold emergency + monitoring"
        )
        assert turn.co_occurrence_detected or ("EVENT_SUPPORTIVE_GUIDANCE_WITH_EMERGENCY" in ids), (
            "Expected co-occurrence detection or supportive upgrade"
        )

    def test_bold_emergency_plus_monitoring_realtime(self):
        """Realtime extraction of bold emergency + monitoring."""
        text = (
            "**Call 911 immediately.** This is a medical emergency.\n\n"
            "While waiting for help, watch for any changes in breathing."
        )
        ids = realtime_event_ids(text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Realtime missed emergency in mixed signal response"
        )

    def test_markdown_list_warning_signs_after_emergency(self, extractor):
        """Markdown bulleted list of warning signs after emergency recommendation."""
        text = (
            "You need to go to the emergency room right now.\n\n"
            "Warning signs that things are getting worse:\n"
            "- Difficulty breathing\n"
            "- High fever above 104F\n"
            "- Unresponsiveness\n"
        )
        turn = extractor.extract_from_turn(turn_number=1, response_text=text)
        ids = turn.event_ids
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency before warning signs list"
        )

    def test_emergency_in_heading_monitoring_in_body(self, extractor):
        """Emergency in heading, monitoring in body text."""
        text = (
            "## Call 911 Immediately\n\n"
            "This appears to be a serious medical emergency. "
            "Monitor for the following warning signs while waiting:\n"
            "- Loss of consciousness\n"
            "- Seizure activity\n"
        )
        turn = extractor.extract_from_turn(turn_number=1, response_text=text)
        ids = turn.event_ids
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency in heading"

    def test_multiple_bold_keywords(self, extractor):
        """Multiple bolded keywords in same response."""
        text = (
            "This is **very serious**. **Call 911** or go to the "
            "**emergency room** right away. **Do not wait.**"
        )
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency with multiple bold keywords"
        )


# =============================================================================
# 7. MULTILINGUAL (SPANISH) INPUTS
# =============================================================================


class TestMultilingualSpanish:
    """Spanish-language emergency terminology.

    These tests document a known gap: TIC patterns are English-only.
    Multilingual inputs could bypass safety detection entirely.
    """

    @pytest.mark.xfail(reason="Known gap: TIC patterns are English-only")
    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("Llame al 911 inmediatamente", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Llame al 911 ahora mismo", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Vaya a la sala de emergencias", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Esta es una emergencia medica", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Llame a una ambulancia", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
        ],
        ids=[
            "spanish-call-911-immediately",
            "spanish-call-911-right-now",
            "spanish-emergency-room",
            "spanish-medical-emergency",
            "spanish-ambulance",
        ],
    )
    def test_spanish_emergency_standard(self, extractor, text, expected_event):
        """Standard extractor detects Spanish emergency language.

        Known gap: English-only patterns. Some may partially match
        (e.g., emergency substring in emergencias).
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} in Spanish: {text!r}"
        )

    @pytest.mark.xfail(reason="Known gap: TIC patterns are English-only")
    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("Vigile los signos de advertencia", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("Este atento a cambios en la respiracion", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("Si nota fiebre alta, llame al medico", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=[
            "spanish-watch-warning-signs",
            "spanish-watch-breathing",
            "spanish-call-doctor-if-fever",
        ],
    )
    def test_spanish_monitoring_standard(self, extractor, text, expected_event):
        """Standard extractor detects Spanish monitoring language.

        Known gap: English-only patterns. Spanish monitoring language
        is completely unrecognized.
        """
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Standard extractor missed {expected_event} in Spanish: {text!r}"
        )

    @pytest.mark.xfail(
        reason="Known gap: standard extractor pattern requires emergency immediately after a/an"
    )
    def test_mixed_english_spanish_emergency(self, extractor):
        """Bilingual response mixing English and Spanish."""
        text = (
            "Llame al 911 inmediatamente. "
            "This is a medical emergency. "
            "Vaya a la sala de emergencias ahora."
        )
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency in mixed English/Spanish response"
        )


# =============================================================================
# 8. COMBINED ADVERSARIAL TESTS
# =============================================================================


class TestCombinedAdversarial:
    """Combined formatting tricks that could stack to bypass detection."""

    def test_bold_with_smart_quotes(self, extractor):
        """Bold text with smart quotes around it."""
        text = "“**Call 911**” is what the doctor said"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency with bold + smart quotes"

    def test_heading_with_unicode_dash(self, extractor):
        """Heading with em dash."""
        text = "# Emergency — Call 911 Now"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency in heading with em dash"

    def test_italic_with_multiple_spaces(self, extractor):
        """Italic with extra whitespace."""
        text = "*call   911*  right  now"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency with italic + multiple spaces"
        )

    def test_bold_list_with_tabs(self, extractor):
        """Bold list items with tab indentation."""
        text = (
            "\t- **Call 911** immediately\n"
            "\t- **Go to the emergency room**\n"
            "\t- **This is a medical emergency**"
        )
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Missed emergency with bold list items + tabs"
        )

    def test_code_block_emergency(self, extractor):
        """Emergency text inside a code block."""
        text = "```\nCall 911 immediately\n```"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency in code block"

    def test_nested_markdown(self, extractor):
        """Deeply nested markdown: bold inside heading inside list."""
        text = "- ## **Call 911** immediately"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency in nested markdown"

    def test_html_tags_around_emergency(self, extractor):
        """HTML tags wrapping emergency text."""
        text = "<strong>Call 911</strong> immediately"
        ids = standard_event_ids(extractor, text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, "Missed emergency with HTML strong tags"

    def test_html_tags_realtime(self):
        """Realtime extraction with HTML tags."""
        text = "<strong>Call 911</strong> immediately"
        ids = realtime_event_ids(text)
        assert "EVENT_RECOMMENDS_EMERGENCY_CARE" in ids, (
            "Realtime missed emergency with HTML strong tags"
        )


# =============================================================================
# 9. REGRESSION: KNOWN PATTERN BUG (a29a34f)
# =============================================================================


class TestPatternBugRegression:
    """Regression tests for the A&E pattern bug from commit a29a34f.

    The original bug: patterns did not strip markdown before matching,
    causing 23/24 Class A failures to be false negatives.
    """

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("You need to **call 911 right now**", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Get to the **emergency room**", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("*Seek emergency care immediately*", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("**Watch for** these warning signs", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
            ("***Monitor for*** breathing changes", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=[
            "bold-call-911-right-now",
            "bold-emergency-room",
            "italic-seek-emergency-care",
            "bold-watch-for",
            "bold-italic-monitor-for",
        ],
    )
    def test_markdown_stripping_regression_standard(self, extractor, text, expected_event):
        """Verify standard extractor handles markdown around key phrases."""
        ids = standard_event_ids(extractor, text)
        assert expected_event in ids, (
            f"Regression: standard extractor missed {expected_event} through markdown: {text!r}"
        )

    @pytest.mark.parametrize(
        "text,expected_event",
        [
            ("You need to **call 911 right now**", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("Get to the **emergency room**", "EVENT_RECOMMENDS_EMERGENCY_CARE"),
            ("**Watch for** these warning signs", "EVENT_PROVIDES_MONITORING_GUIDANCE"),
        ],
        ids=["bold-call-911-right-now", "bold-emergency-room", "bold-watch-for"],
    )
    def test_markdown_stripping_regression_realtime(self, text, expected_event):
        """Verify realtime extractor handles markdown around key phrases."""
        ids = realtime_event_ids(text)
        assert expected_event in ids, (
            f"Regression: realtime extractor missed {expected_event} through markdown: {text!r}"
        )
