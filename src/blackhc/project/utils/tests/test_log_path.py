# Tests for the xpaths package

import pytest

from blackhc.project.utils.log_path import xpath


@pytest.fixture(autouse=True)
def reset_xpath_state():
    # Reset the state of xpath to its initial state
    # This could be resetting variables or re-instantiating the object
    xpath.test_reset()

    yield  # This allows the test to run. After the test, any cleanup can occur if needed.


def test_xpath_no_step():
    assert xpath.metric_name("metric") == "metric"

    # Check all metrics and paths
    assert xpath.all_metrics == {"metric"}
    assert xpath.all_step_names == set()


def test_xpath():
    with xpath.step("outer"):
        with xpath.step("inner"):
            assert xpath.metric_name("metric") == "outer/inner/metric"
        with xpath.step("inner2"):
            assert xpath.metric_name("metric") == "outer/inner2/metric"

    assert xpath.metric_name("metric") == "metric"

    # Check all metrics and paths
    assert xpath.all_metrics == {"metric", "outer/inner/metric", "outer/inner2/metric"}
    assert xpath.all_step_names == {"outer", "outer/inner", "outer/inner2"}


def test_xpath_array():
    with xpath.step("inner"):
        assert xpath.metric_name("metric", is_summary=True) == "inner[0]/metric"
    with xpath.step("inner"):
        assert xpath.metric_name("metric", is_summary=True) == "inner[1]/metric"


def test_xpath_missing_array():
    # Checks that a warning is logged if the same sub_name is used twice without as_array=True
    with pytest.warns(UserWarning):
        with xpath.step("inner", is_summary_step=True):
            pass
        with xpath.step("inner", is_summary_step=True):
            pass

    # Check all metrics and paths
    assert xpath.all_summary_step_names == {"inner", "inner+1"}


def test_xpath_current_step_name():
    with xpath.step("outer"):
        assert xpath.current_step_name == "outer"
        with xpath.step("inner"):
            assert xpath.current_step_name == "outer/inner"
        with xpath.step("inner"):
            assert xpath.current_step_name == "outer/inner"
        assert xpath.current_step_name == "outer"
    assert xpath.current_step_name == ""

    # Check all metrics and paths
    assert xpath.all_step_names == {"outer", "outer/inner"}


def test_xpath_current_unique_path():
    with xpath.step("outer"):
        assert xpath.current_unique_path == "outer[0]"
        with xpath.step("inner"):
            assert xpath.current_unique_path == "outer[0]/inner[0]"
        with xpath.step("inner"):
            assert xpath.current_unique_path == "outer[0]/inner[1]"
        assert xpath.current_unique_path == "outer[0]"
    assert xpath.current_step_name == ""


def test_xpath_all_paths_stats():
    with xpath.step("outer"):
        with xpath.step("inner"):
            with xpath.step("inner2"):
                with xpath.step("inner3"):
                    pass
    assert xpath.all_step_stats["outer/inner/inner2/inner3"].count == 1
    assert xpath.all_step_stats["outer/inner/inner2"].count == 1
    assert xpath.all_step_stats["outer/inner"].count == 1
    assert xpath.all_step_stats["outer"].count == 1

    assert (
        xpath.all_step_stats["outer/inner/inner2/inner3"].mean_time
        == xpath.all_step_stats["outer/inner/inner2/inner3"].total_time
    )
    assert (
        xpath.all_step_stats["outer/inner/inner2"].mean_time
        == xpath.all_step_stats["outer/inner/inner2"].total_time
    )
    assert (
        xpath.all_step_stats["outer/inner"].mean_time
        == xpath.all_step_stats["outer/inner"].total_time
    )
    assert (
        xpath.all_step_stats["outer"].mean_time
        == xpath.all_step_stats["outer"].total_time
    )

    with xpath.step("outer"):
        with xpath.step("inner"):
            with xpath.step("inner2"):
                with xpath.step("inner3"):
                    pass

    assert xpath.all_step_stats["outer/inner/inner2/inner3"].count == 2
    assert xpath.all_step_stats["outer/inner/inner2"].count == 2
    assert xpath.all_step_stats["outer/inner"].count == 2
    assert xpath.all_step_stats["outer"].count == 2

    assert xpath.all_step_stats["outer/inner/inner2/inner3"].mean_time > 0
    assert xpath.all_step_stats["outer/inner/inner2"].mean_time > 0
    assert xpath.all_step_stats["outer/inner"].mean_time > 0
    assert xpath.all_step_stats["outer"].mean_time > 0

    assert xpath.all_step_stats["outer/inner/inner2/inner3"].total_time > 0
    assert xpath.all_step_stats["outer/inner/inner2"].total_time > 0
    assert xpath.all_step_stats["outer/inner"].total_time > 0
    assert xpath.all_step_stats["outer"].total_time > 0


def test_xpath_summary_step():
    with xpath.step("outer", is_summary_step=True):
        with xpath.step("inner"):
            assert xpath.current_step_name == "outer/inner"
            assert xpath.current_unique_path == "outer/inner"
            assert xpath.metric_name("metric") == "outer/inner/metric"
        with pytest.warns(UserWarning):
            with xpath.step("inner"):
                assert xpath.current_step_name == "outer/inner+1"
                assert xpath.current_unique_path == "outer/inner+1"
                assert xpath.metric_name("metric") == "outer/inner+1/metric"
        assert xpath.current_step_name == "outer"
        assert xpath.current_unique_path == "outer"
        assert xpath.metric_name("metric") == "outer/metric"
