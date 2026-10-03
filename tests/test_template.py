from datetime import datetime

import pytest

from homeduplex.prompt import template

NOW = datetime(2026, 10, 2, 18, 30)


def test_variables() -> None:
    assert template.variables("In the {area} ({room}), {now:%H:%M}. {{literal}}") == {"area", "room", "now"}
    assert template.variables("") == set()


@pytest.mark.parametrize("bad", ["{rooms[0]}", "{room.name}", "{0}", "{}", "{unclosed"])
def test_rejects_anything_but_plain_names(bad: str) -> None:
    with pytest.raises(template.TemplateError):
        template.variables(bad)


def test_check_names_unknown_and_available() -> None:
    with pytest.raises(template.TemplateError, match=r"unknown variable \{player\}; available: \{area\}, \{now\}"):
        template.check("{player}", {"area", "now"})
    template.check("{area}", {"area"})


def test_time_of_day_codes() -> None:
    assert template.time_of_day_codes("Today is {now:%A, %B %-d %Y}.") == []
    assert template.time_of_day_codes("{now:%H:%M} {now:%-I %p} {area}") == ["%H", "%M", "%I", "%p"]
    assert template.time_of_day_codes("{room} at noon") == []


def test_render() -> None:
    text = template.render("In the {area} ({room}). It is {now:%A %H:%M}. {{x}}", "office", {"area": "Office"}, NOW)
    assert text == "In the Office (office). It is Friday 18:30. {x}"


def test_render_without_room() -> None:
    assert template.render("[{room}]", None, {}, NOW) == "[]"
