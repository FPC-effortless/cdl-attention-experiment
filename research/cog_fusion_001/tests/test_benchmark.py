"""Cost aggregation and transparent instrumentation checks, without a benchmark."""
import pytest

from fusion.benchmark import distribution, timed_methods, stage_totals


def test_seven_trial_median_and_interquartile_range():
    result = distribution([1, 2, 3, 4, 5, 6, 7])
    assert result["n"] == 7 and result["median"] == 4
    assert result["q1"] == 2.5 and result["q3"] == 5.5 and result["iqr"] == 3
    assert result["samples"] == [1, 2, 3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        distribution([])


def test_timers_preserve_behavior_and_restore_class_methods_after_failure():
    class Counter:
        def add(self, a, b):
            return a + b
    obj = Counter()
    assert "add" not in vars(obj)
    with pytest.raises(RuntimeError, match="fixture"):
        with timed_methods([(obj, "add", "addition")]) as timings:
            assert obj.add(3, 4) == 7
            assert stage_totals(timings)["addition"]["calls"] == 1
            assert stage_totals(timings)["addition"]["wall_seconds"] >= 0
            raise RuntimeError("fixture")
    assert "add" not in vars(obj) and obj.add(3, 4) == 7
