from typing import List, Tuple
import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"\+?\d[\d\s\-()]{7,}\d"),
}


def detect_pii(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    findings: List[Tuple[str, Tuple[int, int]]] = []
    for label, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append((label, (m.start(), m.end())))
    return findings
