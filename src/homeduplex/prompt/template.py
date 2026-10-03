"""Prompt templates: `str.format` syntax with a small, fixed set of variables.

`{room}` is the room id, other names are the room's fields from the config (`{area}`), and `{now}` is the current
local time, formatted with strftime codes: `{now:%A, %B %d}`. Nothing else is available, so a typo is caught when the
config is loaded, not in the middle of a conversation.
"""

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from string import Formatter

BUILTIN_VARIABLES = frozenset({"room", "now"})
# strftime codes that change within a day.
TIME_OF_DAY_CODES = frozenset("HIMSpXcTRrfs")


class TemplateError(ValueError):
    pass


def variables(template: str) -> set[str]:
    """Names a template refers to (`{area}` → "area"; `{now:%H}` → "now")."""
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as e:
        raise TemplateError(f"invalid template: {e}") from None
    names: set[str] = set()
    for _literal, field, _spec, _conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier():
            raise TemplateError(f"{{{field}}}: use plain names such as {{area}}, without indexing or attributes")
        names.add(field)
    return names


def time_of_day_codes(template: str) -> list[str]:
    """The `{now:...}` codes that change during the day, such as %H or %M."""
    found: list[str] = []
    for _literal, field, spec, _conversion in Formatter().parse(template):
        if field == "now" and spec:
            found += [f"%{c}" for c in re.findall(r"%-?(.)", spec) if c in TIME_OF_DAY_CODES]
    return found


def check(template: str, available: Iterable[str]) -> None:
    unknown = variables(template) - set(available)
    if unknown:
        names = ", ".join(f"{{{n}}}" for n in sorted(unknown))
        known = ", ".join(f"{{{n}}}" for n in sorted(available))
        raise TemplateError(f"unknown variable {names}; available: {known}")


def render(template: str, room: str | None, fields: Mapping[str, str], now: datetime) -> str:
    return template.format_map({**fields, "room": room or "", "now": now})
