import pytest

from geoguessr_ai import thermal
from geoguessr_ai.thermal import cpu_temperature, running_cool


def test_reads_the_cpus_temperature_where_it_can():
    first, again = cpu_temperature(), cpu_temperature()
    for reading in (first, again):  # None where there's no thermal zone to read
        assert reading is None or 0 < reading < 130


def test_running_cool_ends_with_what_it_ran_even_when_that_fails(monkeypatch):
    asked = []
    monkeypatch.setattr(thermal, "_efficiency_mode", asked.append)
    with pytest.raises(ValueError), running_cool():
        assert asked == [True]
        raise ValueError
    assert asked == [True, False]


def test_windows_takes_being_asked_to_run_cool():
    with running_cool():
        pass
