def process(symbol: str, extra: str = "") -> None:
    # Dummy function used for tests
    pass


def fetch(symbol: str) -> None:
    # Dummy function used for tests
    pass


def combine(symbol: str, codes: list[str]) -> None:
    # Dummy combine function used for tests
    pass


def expand_codes(symbol: str) -> list[dict[str, str]]:
    """Return two dummy contract codes for expansion tests."""
    _ = symbol
    return [{"code": "AA"}, {"code": "BB"}]
