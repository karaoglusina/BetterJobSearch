"""Benefits/compensation extraction using keyword matching."""

from __future__ import annotations

import re
from typing import List, Set

from ...models.aspect import AspectExtraction

BENEFIT_PATTERNS = [
    # Compensation
    (re.compile(r"\b(competitive\s+salary|salary\s+range|base\s+salary|annual\s+salary)\b", re.I), "salary"),
    (re.compile(r"\b(equity|stock\s+options|shares|esop|rsu|vesting)\b", re.I), "equity"),
    (re.compile(r"\b(bonus|performance\s+bonus|annual\s+bonus|signing\s+bonus)\b", re.I), "bonus"),
    (re.compile(r"\b(profit[- ]?sharing)\b", re.I), "profit sharing"),
    # Time off
    (re.compile(r"\b(\d+)\s*days?\s*(of\s+)?(annual\s+|paid\s+)?(leave|holiday|vacation|pto)\b", re.I), "paid leave"),
    (re.compile(r"\b(unlimited\s+(pto|vacation|leave|time\s+off))\b", re.I), "unlimited PTO"),
    (re.compile(r"\b(paid\s+time\s+off|pto|vacation\s+days?)\b", re.I), "paid leave"),
    (re.compile(r"\b(sabbatical)\b", re.I), "sabbatical"),
    # Health & wellness
    (re.compile(r"\b(health\s+insurance|medical\s+insurance|healthcare)\b", re.I), "health insurance"),
    (re.compile(r"\b(dental|vision)\s+(insurance|coverage|plan)\b", re.I), "dental/vision"),
    (re.compile(r"\b(mental\s+health|wellness\s+program|gym\s+membership|fitness)\b", re.I), "wellness"),
    (re.compile(r"\b(life\s+insurance|disability\s+insurance)\b", re.I), "life insurance"),
    # Retirement
    (re.compile(r"\b(pension|401k|401\(k\)|retirement\s+plan|pensioen)\b", re.I), "pension"),
    # Work flexibility
    (re.compile(r"\b(flexible\s+(hours|schedule|working)|flextime|flexibel)\b", re.I), "flexible hours"),
    (re.compile(r"\b(work[- ]life\s+balance|work\s+life\s+balance)\b", re.I), "work-life balance"),
    (re.compile(r"\b(compressed\s+work\s*week|4[- ]day\s+week|32[- ]hour)\b", re.I), "compressed week"),
    # Development
    (re.compile(r"\b(training\s+budget|learning\s+budget|education\s+budget|development\s+budget)\b", re.I), "training budget"),
    (re.compile(r"\b(professional\s+development|career\s+(development|growth)|l&d)\b", re.I), "career development"),
    (re.compile(r"\b(conference|courses?|certifications?|workshops?)\b", re.I), "learning opportunities"),
    # Equipment & office
    (re.compile(r"\b(laptop|macbook|equipment\s+(budget|allowance))\b", re.I), "equipment"),
    (re.compile(r"\b(home\s+office\s+(budget|allowance|setup))\b", re.I), "home office budget"),
    # Transport & food
    (re.compile(r"\b(travel\s+allowance|commute|transport\s+allowance|ns[- ]business\s+card|ov[- ]?chipkaart)\b", re.I), "transport"),
    (re.compile(r"\b(lunch|meals?|snacks|free\s+food|catering)\b", re.I), "meals"),
    (re.compile(r"\b(company\s+car|lease\s+car|car\s+allowance)\b", re.I), "company car"),
    # Other
    (re.compile(r"\b(relocation\s+(support|package|assistance))\b", re.I), "relocation"),
    (re.compile(r"\b(parental\s+leave|maternity|paternity)\b", re.I), "parental leave"),
    (re.compile(r"\b(employee\s+discount|corporate\s+discount)\b", re.I), "discounts"),
    (re.compile(r"\b(team\s+(events?|building|outings?)|company\s+events?|borrels?)\b", re.I), "team events"),
]


class BenefitsExtractor:
    """Extract benefits and compensation info from job descriptions."""

    def extract(self, text: str) -> AspectExtraction:
        values: List[str] = []
        spans: List[str] = []
        seen: Set[str] = set()

        for pattern, benefit in BENEFIT_PATTERNS:
            for match in pattern.finditer(text):
                if benefit not in seen:
                    seen.add(benefit)
                    values.append(benefit)
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    spans.append(text[start:end].strip())
                    break

        return AspectExtraction(
            aspect="benefits",
            values=values,
            evidence_spans=spans,
            confidence=1.0 if values else 0.0,
            method="regex",
        )
