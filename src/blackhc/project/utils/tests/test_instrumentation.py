from blackhc.project.utils.instrumentation import (
    Context,
    Instrumentation,
    NullInstrumentation,
)


def test_instrumentation_example():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)
            assert instrument.spy(loss=3) == 3

    assert run_info == Context(loss=[1], my_scope=[Context(info=[2], loss=[3])])


def test_instrumentation_decorator():
    instrument = Instrumentation()

    @instrument.scope("my_scope")
    def my_func():
        instrument.log(info=2)

    with instrument.collect() as run_info:
        instrument.log(loss=1)
        my_func()

    assert run_info == Context(loss=[1], my_scope=[Context(info=[2])])


def test_instrumentation_nested_scopes():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)

            with instrument.scope("my_scope"):
                instrument.log(info=3)

    assert run_info == Context(
        loss=[1],
        my_scope=[
            Context(
                info=[2],
                my_scope=[
                    Context(info=[3]),
                ],
            ),
        ],
    )


def test_instrumentation_nested_collects():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.collect() as run_info_inner:
            instrument.log(loss=2)

            with instrument.scope("my_scope"):
                instrument.log(info=3)

    assert run_info == Context(
        loss=[1, 2],
        my_scope=[
            Context(info=[3]),
        ],
    )

    assert run_info_inner == Context(
        loss=[2],
        my_scope=[
            Context(info=[3]),
        ],
    )


def test_instrumentation_walk():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)

    assert list(run_info.walk()) == [
        ("loss", 1),
        ("my_scope", Context(info=[2])),
    ]


def test_instrumentation_tree_walk():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)

    assert list(run_info.tree_walk()) == [
        (("loss",), 1),
        (("my_scope",), Context(info=[2])),
        (("my_scope", "info"), 2),
    ]

    # Check that we can pattern match as well
    for item in run_info.tree_walk():
        match item:
            case (["loss"], value):
                assert value == 1
            case (["my_scope"], context):
                assert context == Context(info=[2])
            case (["my_scope", "info"], value):
                assert value == 2


def test_instrumentation_to_plain_dict():
    instrument = Instrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)

    print(run_info)

    assert run_info.to_dict() == {
        "loss": [1],
        "my_scope": [
            {
                "info": [2],
            }
        ],
    }


def test_null_instrumentation():
    instrument = NullInstrumentation()

    with instrument.collect() as run_info:
        instrument.log(loss=1)

        with instrument.scope("my_scope"):
            instrument.log(info=2)

    assert run_info == Context()


if __name__ == "__main__":
    test_instrumentation_example()
    test_instrumentation_decorator()
    test_instrumentation_nested_scopes()
    test_instrumentation_nested_collects()
    test_instrumentation_walk()
    test_instrumentation_tree_walk()
