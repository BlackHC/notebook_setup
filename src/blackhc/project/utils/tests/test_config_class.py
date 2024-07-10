# Tests for config_class.
import pytest

from blackhc.project.utils.config_class import configclass


@configclass
class DummyConfigClass:
    a: int
    b: str
    c: float


def test_config_class():
    config = DummyConfigClass(a=1, b="2", c=3.0)
    assert config.a == 1
    assert config.b == "2"
    assert config.c == 3.0

    assert config["a"] == 1
    assert config["b"] == "2"
    assert config["c"] == 3.0


def test_config_class_conversions():
    config = DummyConfigClass(a=1, b="2", c=3.0)

    assert dict(config) == {"a": 1, "b": "2", "c": 3.0}
    assert list(config) == [1, "2", 3.0]
    assert tuple(config) == (1, "2", 3.0)
    assert set(config) == {1, "2", 3.0}
    assert list(config.keys()) == ["a", "b", "c"]
    assert list(config.values()) == [1, "2", 3.0]
    assert list(config.items()) == [("a", 1), ("b", "2"), ("c", 3.0)]


def test_config_class_default_value():
    # throws exception if we don't pass c
    with pytest.raises(ValueError):
        DummyConfigClass(a=1, b="2")

    DummyConfigClass.c = 4.0

    config = DummyConfigClass(a=1, b="2")

    assert config.c == 4.0


def test_config_class_merge_via_or():
    config = DummyConfigClass(a=1, b="2", c=3.0)
    config2 = DummyConfigClass(a=2, b="3", c=4.0)

    config3 = config | config2

    assert config3.a == 2
    assert config3.b == "3"
    assert config3.c == 4.0

    # test merging with a dict
    config4 = config | {"a": 3, "b": "4", "c": 5.0}

    assert config4.a == 3
    assert config4.b == "4"
    assert config4.c == 5.0


def test_config_class_unpacking():
    config = DummyConfigClass(a=1, b="2", c=3.0)

    a, b, c = config

    assert a == 1
    assert b == "2"
    assert c == 3.0

    assert {**config} == {"a": 1, "b": "2", "c": 3.0}


def test_config_class_invoke():
    @configclass
    class DummyConfigClass:
        a: int
        b: str
        c: float

    def func(a, b, c):
        return a, b, c

    config = DummyConfigClass(a=1, b="2", c=3.0)
    result = config.invoke(func)
    assert result == (1, "2", 3.0)


def test_config_class_invoke_kwargs():
    @configclass
    class DummyConfigClass:
        a: int
        b: str
        c: float

    def func(**kwargs):
        return kwargs["a"], kwargs["b"], kwargs["c"]

    config = DummyConfigClass(a=1, b="2", c=3.0)
    result = config.invoke(func)
    assert result == (1, "2", 3.0)
