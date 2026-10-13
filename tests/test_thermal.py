from geoguessr_ai.thermal import cpu_temperature


def test_reads_the_cpus_temperature_where_it_can():
    first, again = cpu_temperature(), cpu_temperature()
    for reading in (first, again):  # None where there's no thermal zone to read
        assert reading is None or 0 < reading < 130
