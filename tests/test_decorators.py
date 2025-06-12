import threading
import time
import pytest

from ifera.decorators import singleton, ThreadSafeCache


def test_singleton_single_thread():
    class Counter:
        count = 0

    @singleton
    class Dummy:
        def __init__(self, value: int) -> None:
            Counter.count += 1
            self.value = value

    first = Dummy(1)
    second = Dummy(2)

    assert first is second
    assert first.value == 1
    assert Counter.count == 1


def test_singleton_multi_thread():
    class Counter:
        count = 0

    @singleton
    class Dummy:
        def __init__(self, value: int) -> None:
            time.sleep(0.05)
            Counter.count += 1
            self.value = value

    results: list[Dummy] = []
    barrier = threading.Barrier(5)

    def creator():
        barrier.wait()
        results.append(Dummy(1))

    threads = [threading.Thread(target=creator) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(id(r) for r in results)) == 1
    assert results[0].value == 1
    assert Counter.count == 1


def test_thread_safe_cache_single_thread():
    calls = {"count": 0}

    @ThreadSafeCache()
    def compute(x: int) -> int:
        calls["count"] += 1
        return x * 2

    assert compute(2) == 4
    assert compute(2) == 4
    assert compute(x=2) == 4
    assert calls["count"] == 1

    assert compute(3) == 6
    assert calls["count"] == 2


def test_thread_safe_cache_multi_thread():
    calls = {"count": 0}

    @ThreadSafeCache()
    def compute(x: int) -> int:
        calls["count"] += 1
        return x * 2

    barrier = threading.Barrier(5)
    results: list[int] = []

    def caller():
        barrier.wait()
        results.append(compute(5))

    threads = [threading.Thread(target=caller) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1)
        if t.is_alive():
            pytest.fail("Deadlock detected in ThreadSafeCache")

    assert len(results) == 5
    assert all(r == 10 for r in results)
    assert calls["count"] == 1
