from global_land_mask import globe

from geoguessr_ai.random_guess import LATITUDES, RandomGuesser


def test_guesses_somewhere_new_on_land_each_time():
    guesser = RandomGuesser(seed=1)
    guesses = [guesser.predict([]) for _ in range(200)]

    assert all(globe.is_land(g.lat, g.lon) for g in guesses)
    assert all(LATITUDES[0] <= g.lat <= LATITUDES[1] for g in guesses)
    assert len({(g.lat, g.lon) for g in guesses}) == 200
    assert min(g.lon for g in guesses) < -60 and max(g.lon for g in guesses) > 60  # all over
    assert all(g.expected_score == 0 for g in guesses)
