"""Remote/hybrid/onsite policy extraction using regex patterns."""

from __future__ import annotations

import re

from ...models.aspect import AspectExtraction

REMOTE_PATTERNS = [
    # Fully remote
    (re.compile(r"\b(fully|100%|completely)\s+remote\b", re.I), "remote"),
    (re.compile(r"\bremote[- ]first\b", re.I), "remote"),
    (re.compile(r"\bwork\s+(from\s+)?(home|anywhere)\b", re.I), "remote"),
    (re.compile(r"\bremote\s+(position|role|job|work|opportunity)\b", re.I), "remote"),
    # Hybrid
    (re.compile(r"\bhybrid\s+(work|model|role|position|environment|setup|arrangement)\b", re.I), "hybrid"),
    (re.compile(r"\b(hybrid)\b(?!\s+cloud)", re.I), "hybrid"),
    (re.compile(r"\b(\d+)\s*days?\s*(in\s+)?(the\s+)?office\b", re.I), "hybrid"),
    (re.compile(r"\bcombination\s+of\s+(remote|office|home)\b", re.I), "hybrid"),
    (re.compile(r"\b(remote|home)\s+and\s+(office|on[- ]?site)\b", re.I), "hybrid"),
    # Onsite
    (re.compile(r"\b(on[- ]?site|in[- ]?office|office[- ]based)\b", re.I), "onsite"),
    (re.compile(r"\b(fully|100%)\s+(on[- ]?site|in[- ]?office)\b", re.I), "onsite"),
    (re.compile(r"\bpresence\s+(in|at)\s+(the\s+)?office\b", re.I), "onsite"),
]

# Priority order: more specific wins
PRIORITY = {"remote": 3, "hybrid": 2, "onsite": 1}


class RemotePolicyExtractor:
    """Extract remote work policy from job descriptions."""

    def extract(self, text: str) -> AspectExtraction:
        matches: dict[str, str] = {}  # policy -> evidence

        for pattern, policy in REMOTE_PATTERNS:
            match = pattern.search(text)
            if match and policy not in matches:
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                matches[policy] = text[start:end].strip()

        if not matches:
            return AspectExtraction(
                aspect="remote_policy",
                values=[],
                evidence_spans=[],
                confidence=0.0,
                method="regex",
            )

        # If multiple policies detected, pick highest priority
        best = max(matches.keys(), key=lambda p: PRIORITY.get(p, 0))

        return AspectExtraction(
            aspect="remote_policy",
            values=[best],
            evidence_spans=[matches[best]],
            confidence=1.0,
            method="regex",
        )
