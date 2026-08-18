"""Serbest metin kullanici girdisini yapilandirilmis filtreye cevirir (Omer).

MVP: regex/anahtar kelime tabanli, API bagimliligi yok.
Stretch: LLM API ile ayni ciktiyi (JSON) uretmek - regex surumu her zaman yedek kalir.
"""

from dataclasses import dataclass, field


@dataclass
class UserProfile:
    budget_max: float | None = None
    room_options: list[str] = field(default_factory=list)
    target_district: str | None = None
    alternative_districts: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    raw_text: str = ""


def parse_budget(text: str) -> float | None:
    raise NotImplementedError


def parse_room_count(text: str) -> list[str]:
    raise NotImplementedError


def parse_districts(text: str, known_districts: list[str]) -> tuple[str | None, list[str]]:
    raise NotImplementedError


def parse_priorities(text: str) -> list[str]:
    raise NotImplementedError


def parse_user_input(text: str, known_districts: list[str]) -> UserProfile:
    raise NotImplementedError
