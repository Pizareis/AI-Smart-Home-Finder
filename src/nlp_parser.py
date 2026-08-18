"""Serbest metin kullanici girdisini yapilandirilmis filtreye cevirir (Omer).

MVP: regex/anahtar kelime tabanli, API bagimliligi yok, demo her zaman calisir.
Stretch: LLM API ile ayni ciktiyi (JSON) uretmek - regex surumu her zaman yedek kalir.
"""

import re
from dataclasses import dataclass, field

# Yagmur'un dataset calismasi bitene kadar gecici placeholder liste.
# Gercek mahalle listesi data/processed/district_stats.csv ile degistirilecek.
KNOWN_DISTRICTS = [
    "Görükle", "Beşevler", "Özlüce", "Nilüfer", "İhsaniye",
    "Fethiye", "Ertuğrul", "Konak", "Ataevler", "Yıldırım",
]

PRIORITY_KEYWORDS = {
    "transport": ["ulaşım", "ulasim", "toplu taşıma", "toplu tasima", "otobüs", "otobus", "metro"],
    "safety": ["güvenlik", "guvenlik", "güvenli", "guvenli"],
    "distance": ["okula yakın", "okula yakin", "işe yakın", "ise yakin", "üniversiteye yakın", "universiteye yakin", "mesafe"],
    "social": ["sosyal", "kafe", "eğlence", "eglence"],
    "budget": ["ucuz", "uygun fiyat", "bütçe", "butce"],
    "features": ["eşyalı", "esyali", "yeni bina", "asansör", "asansor"],
}

_BUDGET_THOUSAND_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*bin\s*(?:tl|lira)?", re.IGNORECASE)
_BUDGET_PLAIN_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:tl|lira)", re.IGNORECASE)
_ROOM_RE = re.compile(r"\b(\d)\s*\+\s*(\d)\b")
_STUDIO_RE = re.compile(r"\b(stüdyo|studyo|studio)\b", re.IGNORECASE)


@dataclass
class UserProfile:
    budget_max: float | None = None
    room_options: list[str] = field(default_factory=list)
    target_district: str | None = None
    alternative_districts: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    raw_text: str = ""


def parse_budget(text: str) -> float | None:
    match = _BUDGET_THOUSAND_RE.search(text)
    if match:
        return float(match.group(1).replace(",", ".")) * 1000

    match = _BUDGET_PLAIN_RE.search(text)
    if match:
        return float(match.group(1).replace(".", "").replace(",", ""))

    return None


def parse_room_count(text: str) -> list[str]:
    options = [f"{a}+{b}" for a, b in _ROOM_RE.findall(text)]
    if _STUDIO_RE.search(text):
        options.append("studio")
    return options


def parse_districts(text: str, known_districts: list[str] = KNOWN_DISTRICTS) -> tuple[str | None, list[str]]:
    lowered = text.lower()
    found = []
    for district in known_districts:
        pos = lowered.find(district.lower())
        if pos != -1:
            found.append((pos, district))

    found.sort(key=lambda item: item[0])
    ordered = [district for _, district in found]

    if not ordered:
        return None, []

    return ordered[0], ordered[1:]


def parse_priorities(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for tag, keywords in PRIORITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                found.append(tag)
                break
    return found


def parse_user_input(text: str, known_districts: list[str] = KNOWN_DISTRICTS) -> UserProfile:
    target, alternatives = parse_districts(text, known_districts)
    return UserProfile(
        budget_max=parse_budget(text),
        room_options=parse_room_count(text),
        target_district=target,
        alternative_districts=alternatives,
        priorities=parse_priorities(text),
        raw_text=text,
    )
