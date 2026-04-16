from app import greet


def test_greet_matches_expected_output() -> None:
    assert greet("Hans") == "Hello, Hans!"
