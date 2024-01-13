from blackhc.project.utils.tree_namespace import TreeNamespace, items, keys, schema, values


def test_tree_namespace_simple():
    # Create a dictionary for testing
    test_dict = {"a/b": 1, "a/c": 2, "d": 3}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test __getitem__ method
    assert dkns["a/b"] == 1, "Failed on __getitem__ method"
    assert dkns["a/c"] == 2, "Failed on __getitem__ method"
    assert dkns["d"] == 3, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(dkns) == 2, "Failed on __len__ method"
    # Test __repr__ method
    assert (
        repr(dkns) == "TreeNamespace({'a': {'b': 1, 'c': 2}, 'd': 3})"
    ), "Failed on __repr__ method:\n" + repr(dkns)

    assert dkns.a.b == 1, "Failed on __getattribute__ method"
    assert dkns.a.c == 2, "Failed on __getattribute__ method"
    assert dkns.d == 3, "Failed on __getattribute__ method"


def test_tree_namespace_path():
    # Create a dictionary for testing
    test_dict = {"a/c": 2, "d": 3, "a/b/c": 4, "a/b/d": 5}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test __getitem__ method with a sub-path
    sub_dkns = dkns["a/b"]
    assert sub_dkns["c"] == 4, "Failed on __getitem__ method"
    assert sub_dkns["d"] == 5, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(sub_dkns) == 2, "Failed on __len__ method"


def test_tree_namespace_wildcard():
    # Create a dictionary for testing
    test_dict = {"a/c": 2, "d": 3, "a/b/c": 4, "a/b/d": 5}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test __getitem__ method with a wildcard
    sub_dkns = dkns["a/*"]
    assert sub_dkns["c"] == 2, "Failed on __getitem__ method"
    assert sub_dkns["b/c"] == 4, "Failed on __getitem__ method"
    assert sub_dkns["b/d"] == 5, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(sub_dkns) == 2, "Failed on __len__ method"


def test_tree_namespace_wildcard_2():
    # Create a dictionary for testing with two sub-arrays: a/0/0 etc
    test_dict = {"a/0/0": 1, "a/0/1": 2, "a/1/0": 3, "a/1/1": 4}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test __getitem__ method with a wildcard
    sub_dkns = dkns["a/*/0"]
    assert sub_dkns["0"] == 1, "Failed on __getitem__ method"
    assert sub_dkns["1"] == 3, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(sub_dkns) == 2, "Failed on __len__ method"

    # Test a double wildcard
    sub_dkns = dkns["a/*/*"]
    assert sub_dkns["0/0"] == 1, "Failed on __getitem__ method"
    assert sub_dkns["0/1"] == 2, "Failed on __getitem__ method"
    assert sub_dkns["1/0"] == 3, "Failed on __getitem__ method"
    assert sub_dkns["1/1"] == 4, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(sub_dkns) == 2, "Failed on __len__ method"
    # Test sub access
    assert sub_dkns[0][0] == 1, "Failed on __getitem__ method"
    assert sub_dkns[0][1] == 2, "Failed on __getitem__ method"
    assert sub_dkns[1][0] == 3, "Failed on __getitem__ method"
    assert sub_dkns[1][1] == 4, "Failed on __getitem__ method"


def test_tree_namespace_wildcard_at():
    # Create a dictionary for testing with a nested key of interest (loss) at different nesting levels
    test_dict = {
        "a/0/0/loss": 1,
        "a/0/1/loss": 2,
        "a/1/0/loss": 3,
        "a/1/1/loss": 4,
        "a/loss": 5,
    }
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test __getitem__ method with a wildcard
    sub_dkns = dkns["@/loss"]
    assert sub_dkns["a_0_0"] == 1, "Failed on __getitem__ method"
    assert sub_dkns["a_0_1"] == 2, "Failed on __getitem__ method"
    assert sub_dkns["a_1_0"] == 3, "Failed on __getitem__ method"
    assert sub_dkns["a_1_1"] == 4, "Failed on __getitem__ method"
    assert sub_dkns["a"] == 5, "Failed on __getitem__ method"
    # Test __len__ method
    assert len(sub_dkns) == 5, "Failed on __len__ method"


def test_tree_namespace_keys():
    # Create a dictionary for testing
    test_dict = {"a/b": 1, "a/c": 2, "d": 3}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test keys method
    assert keys(dkns) == ["a", "d"], "Failed on keys method"


def test_tree_namespace_values():
    # Create a dictionary for testing
    test_dict = {"a/b": 1, "a/c": 2, "d": 3}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test values method
    assert values(dkns) == [dkns.a, dkns.d], f"Failed on values method:\n{values(dkns)}"


def test_tree_namespace_items():
    # Create a dictionary for testing
    test_dict = {"a/b": 1, "a/c": 2, "d": 3}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test items method
    assert items(dkns) == [("a", dkns.a), ("d", dkns.d)], "Failed on items method"


def test_tree_namespace_schema():
    # Create a dictionary for testing
    test_dict = {"a/b": 1, "a/c": 2, "d": 3}
    # Create a DeepKeyNamespace object
    dkns = TreeNamespace(test_dict)
    # Test schema method
    assert schema(dkns) == {"a/b": int, "a/c": int, "d": int}, "Failed on schema method"
    assert schema(dkns) == {
        "a": {"b": int, "c": int},
        "d": int,
    }, "Failed on schema method"


if __name__ == "__main__":
    test_tree_namespace_simple()
    test_tree_namespace_path()
    test_tree_namespace_wildcard()
    test_tree_namespace_wildcard_2()
    test_tree_namespace_wildcard_at()
    test_tree_namespace_keys()
    test_tree_namespace_values()
    test_tree_namespace_items()
    test_tree_namespace_schema()
