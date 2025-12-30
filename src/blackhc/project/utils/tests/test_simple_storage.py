from pyfakefs import fake_filesystem
from blackhc.project.utils import simple_storage
from blackhc.project.utils.simple_storage import (
    Storage,
    prefix_schema,
    identify,
    load_from_path,
    load_metadata_from_path,
    ObjectIdentityRegistry,
    _format_value,
    _format_kvs,
    _collect_metadata,
)
import pytest
import enum


class _TestObject:
    """A test object that supports weakrefs."""

    def __init__(self, value):
        self.value = value


# =============================================================================
# Value Formatting Tests
# =============================================================================


def test_format_value():
    assert _format_value(123) == "123"
    assert _format_value(123.456) == "123.456"
    assert _format_value("test_string") == "test_string"
    assert _format_value([1, 2, 3]) == "[1,2,3]"
    assert _format_value((1, 2, 3)) == "(1,2,3)"
    assert _format_value({"key": "value"}) == "{key:value}"
    assert _format_value({"key": True}) == "{+key}"
    assert _format_value({"key": False}) == "{-key}"
    assert (
        _format_value({None: "value"}) == "{None:value}"
    )  # None key is formatted as "None"
    assert _format_value(123.0) == "123"
    assert _format_value(["a", "b", "c"]) == "[a,b,c]"
    assert _format_value({"a": 1, "b": 2}) == "{a:1,b:2}"
    # Sets are sorted for deterministic output
    assert _format_value({3, 1, 2}) == "{1,2,3}"
    assert _format_value({"c", "a", "b"}) == "{a,b,c}"
    assert _format_value(frozenset([3, 1, 2])) == "{1,2,3}"
    assert _format_value(set()) == "{}"

    class TestEnum(enum.Enum):
        OPTION_A = "OptionA"
        OPTION_B = "OptionB"

    assert _format_value(TestEnum.OPTION_A) == "OptionA"
    assert _format_value(TestEnum.OPTION_B) == "OptionB"


def test_format_value_unknown_type():
    with pytest.raises(TypeError, match="Cannot format value"):
        _format_value(object())


def test_format_value_with_int_enum():
    class IntEnum(enum.Enum):
        OPTION_1 = 1
        OPTION_2 = 2

    assert _format_value(IntEnum.OPTION_1) == "OPTION_1"
    assert _format_value(IntEnum.OPTION_2) == "OPTION_2"


def test_format_kwargs_fragment():
    assert _format_kvs({"key": "value"}) == "key:value"
    assert (
        _format_kvs({"key1": "value1", "key2": "value2"})
        == "key1:value1_key2:value2"
    )
    assert _format_kvs({"key": 123}) == "key:123"
    assert _format_kvs({"key": None}) == "~key"
    assert _format_kvs({"key": 123.0}) == "key:123"
    # None key is special-cased to just show the value
    assert _format_kvs({None: "value"}) == "value"
    assert _format_kvs({None: 123}) == "123"


# =============================================================================
# Object Identity Tests
# =============================================================================


def test_object_identity_registry():
    registry = ObjectIdentityRegistry()

    obj = _TestObject([1, 2, 3])
    registry.identify(obj, "my_list")

    assert obj in registry
    assert registry.get(obj) == "my_list"
    assert registry.resolve(obj) == "my_list"
    assert registry[obj] == "my_list"

    # Unknown object
    other = _TestObject([4, 5, 6])
    assert other not in registry
    assert registry.get(other) is None
    assert registry.get(other, "default") == "default"
    assert registry.resolve(other) == other


def test_global_identify():
    obj = _TestObject({"test": "object"})
    result = identify(obj, "test_obj")

    assert result is obj
    assert obj in simple_storage.object_identities
    assert simple_storage.object_identities.get(obj) == "test_obj"


# =============================================================================
# Storage Class Tests
# =============================================================================


def test_storage_save_and_load_pkl(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    test_obj = {"key": "value"}

    path, metadata = storage.save(test_obj, "test", fmt="pkl")
    loaded_obj = storage.load("test")

    assert test_obj == loaded_obj
    assert (path.parent / "meta.json").exists()
    assert path.exists()  # path is the data file itself


def test_storage_save_and_load_json(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    test_obj = {"key": "value"}

    path, metadata = storage.save(test_obj, "test", fmt="json")
    loaded_obj = storage.load("test")

    assert test_obj == loaded_obj
    assert (path.parent / "meta.json").exists()
    assert path.exists()  # path is the data file itself


def test_storage_save_and_load_pt(tmp_path):
    # Use real filesystem (tmp_path) because torch.save uses C++ code
    # that bypasses Python's file layer, making pyfakefs incompatible
    if simple_storage.torch is None:
        pytest.skip("torch is not available")

    storage = Storage(tmp_path, verbose=False)
    test_obj = simple_storage.torch.tensor([1, 2, 3])

    path, metadata = storage.save(test_obj, "test", fmt="pt")
    loaded_obj = storage.load("test")

    assert simple_storage.torch.equal(test_obj, loaded_obj)
    assert (path.parent / "meta.json").exists()
    assert path.exists()  # path is the data file itself


def test_storage_auto_format(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    # Small JSON-compatible object should use JSON
    small_obj = {"key": "value"}
    path, _ = storage.save(small_obj, "small")
    assert path.suffix == ".json"

    # Complex object should use pickle (set is not JSON serializable)
    complex_obj = {"key": {1, 2, 3}}
    path, _ = storage.save(complex_obj, "complex")
    assert path.suffix == ".pkl"


def test_storage_versioning(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    # Save with version=True creates timestamped subdirectory
    path1, _ = storage.save({"v": 1}, "test", version=True)
    path2, _ = storage.save({"v": 2}, "test", version=True)

    assert path1 != path2
    # Both should be under the same base path (test/), their parent.parent is the same
    assert path1.parent.parent == path2.parent.parent

    # Load latest (version=True means latest for loading)
    latest = storage.load("test", version=True)
    assert latest == {"v": 2}

    # Load specific version (path1.parent.name is the version timestamp directory)
    specific = storage.load("test", version=path1.parent.name)
    assert specific == {"v": 1}


def test_storage_with_dict_parts(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    test_obj = {"result": 42}

    path, _ = storage.save(test_obj, "experiments", {"model": "cnn", "epochs": 10})
    loaded = storage.load("experiments", {"model": "cnn", "epochs": 10})

    assert test_obj == loaded


def test_storage_exists(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    assert not storage.exists("nonexistent")

    storage.save({"key": "value"}, "exists_test")
    assert storage.exists("exists_test")


def test_storage_prepare_output(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    output_path, metadata = storage.prepare_output("output_test", fmt="pkl")

    assert output_path.name == "data.pkl"
    assert output_path.parent.exists()


def test_storage_load_metadata(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    storage.save({"key": "value"}, "meta_test")

    metadata = storage.load_metadata("meta_test")

    assert metadata.timestamp is not None
    assert metadata.path_parts == ["meta_test"]


def test_storage_load_all_metadata(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    storage.save({"a": 1}, "exp1")
    storage.save({"b": 2}, "exp2")

    all_meta = storage.load_all_metadata()

    assert len(all_meta) == 2


# =============================================================================
# Verbose and Format Detection Tests
# =============================================================================


def test_storage_verbose_prints(fs: fake_filesystem.FakeFilesystem, capsys):
    """Test that verbose=True prints cache messages."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=True)

    @storage.cache(prefix_schema())
    def compute():
        return 42

    # First call should print "Cached to"
    compute()
    captured = capsys.readouterr()
    assert "📦 Cached to" in captured.out

    # Second call should print "Loaded from cache"
    compute()
    captured = capsys.readouterr()
    assert "📦 Loaded from cache" in captured.out


def test_storage_verbose_false_no_prints(fs: fake_filesystem.FakeFilesystem, capsys):
    """Test that verbose=False suppresses cache messages."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    @storage.cache(prefix_schema())
    def compute():
        return 42

    compute()
    compute()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_detect_format_json(fs: fake_filesystem.FakeFilesystem):
    """Test _detect_format chooses JSON for small JSON-serializable objects."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    fmt, pickled = storage._detect_format({"key": "value"})
    assert fmt == "json"
    assert pickled is not None  # Should return pre-pickled bytes


def test_detect_format_pickle_for_sets(fs: fake_filesystem.FakeFilesystem):
    """Test _detect_format chooses pickle for non-JSON-serializable objects."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    fmt, pickled = storage._detect_format({1, 2, 3})  # Sets aren't JSON serializable
    assert fmt == "pkl"
    assert pickled is not None


def test_detect_format_pickle_for_large_objects(fs: fake_filesystem.FakeFilesystem):
    """Test _detect_format chooses pickle for large objects even if JSON-compatible."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    # Create a large JSON-compatible object (> 256KB)
    large_obj = {"data": "x" * (300 * 1024)}
    fmt, pickled = storage._detect_format(large_obj)
    assert fmt == "pkl"
    assert pickled is not None


def test_detect_format_torch_tensor(tmp_path):
    """Test _detect_format chooses pt for torch tensors."""
    # Use real filesystem because torch C++ code bypasses pyfakefs
    if simple_storage.torch is None:
        pytest.skip("torch is not available")

    storage = Storage(tmp_path, verbose=False)

    tensor = simple_storage.torch.tensor([1, 2, 3])
    fmt, pickled = storage._detect_format(tensor)
    assert fmt == "pt"
    assert pickled is None  # No pre-serialization for torch


# =============================================================================
# Cache Decorator Tests
# =============================================================================


def test_storage_cache_decorator(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    counter = 0

    @storage.cache(prefix_schema("arg1"), fmt="json")
    def test_function(arg1, arg2):
        nonlocal counter
        counter += 1
        return {"result": arg1 + arg2}

    # Test loading directly from cache fails when empty
    with pytest.raises(FileNotFoundError):
        test_function.load(1, 2)

    assert counter == 0

    # First call computes
    result = test_function(1, 2)
    assert result == {"result": 3}
    assert counter == 1

    # Second call uses cache
    cached_result = test_function(1, 2)
    assert cached_result == result
    assert counter == 1

    # Load directly from cache
    loaded_result = test_function.load(1, 2)
    assert loaded_result == result
    assert counter == 1

    # Recompute
    recomputed_result = test_function.recompute(1, 2)
    assert recomputed_result == result
    assert counter == 2

    # Should have two version subdirectories now
    prefix_path = test_function.get_prefix_path(1, 2)
    assert len(list(prefix_path.iterdir())) == 2


def test_storage_cache_with_remaining_args(fs: fake_filesystem.FakeFilesystem):
    """Test that remaining args are automatically included in cache path."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    @storage.cache(prefix_schema("dataset"))
    def train(dataset, model, epochs=10):
        return {"model": model, "epochs": epochs}

    # Different epochs should cache separately
    result1 = train("mnist", "cnn", epochs=10)
    result2 = train("mnist", "cnn", epochs=20)

    assert result1 != result2

    # Verify by loading (calling again should use cache)
    loaded1 = train("mnist", "cnn", epochs=10)
    loaded2 = train("mnist", "cnn", epochs=20)

    assert loaded1 == result1
    assert loaded2 == result2


def test_storage_cache_with_object_identity(fs: fake_filesystem.FakeFilesystem):
    """Test that object identities are resolved in cache paths."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    registry = ObjectIdentityRegistry()
    storage = Storage(root, identity_registry=registry, verbose=False)

    model_obj = _TestObject({"type": "resnet", "layers": 50})
    registry.identify(model_obj, "resnet50")

    @storage.cache(prefix_schema("model"))
    def evaluate(model, dataset):
        return {"accuracy": 0.95}

    result = evaluate(model_obj, "imagenet")

    # The path should use "resnet50" not the object representation
    path = evaluate.get_prefix_path(model_obj, "imagenet")
    assert "resnet50" in str(path)


def test_cache_force_refresh(fs: fake_filesystem.FakeFilesystem):
    """Test _force_refresh parameter."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    counter = 0

    @storage.cache(prefix_schema())
    def compute():
        nonlocal counter
        counter += 1
        return counter

    result1 = compute()
    assert result1 == 1
    assert counter == 1

    # Should use cache
    result2 = compute()
    assert result2 == 1
    assert counter == 1

    # Force refresh should recompute
    result3 = compute(_force_refresh=True)
    assert result3 == 2
    assert counter == 2


def test_cache_load_with_version(fs: fake_filesystem.FakeFilesystem):
    """Test .load() with _version parameter."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    @storage.cache(prefix_schema("x"))
    def compute(x):
        return x * 2

    # Create two versions
    result1 = compute(5)
    assert result1 == 10

    # Get the first version's timestamp from path
    path1 = compute.get_prefix_path(5)
    versions = sorted(d.name for d in path1.iterdir() if d.is_dir())
    assert len(versions) == 1
    first_version = versions[0]

    # Force a second version
    result2 = compute(5, _force_refresh=True)
    assert result2 == 10

    versions = sorted(d.name for d in path1.iterdir() if d.is_dir())
    assert len(versions) == 2

    # Load specific version by name
    loaded = compute.load(5, _version=first_version)
    assert loaded == 10

    # Load latest (default)
    loaded_latest = compute.load(5)
    assert loaded_latest == 10


def test_collect_metadata(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/blackhc.project"
    fs.create_dir(root)

    metadata = _collect_metadata(["test"])
    assert metadata.timestamp is not None
    assert metadata.path_parts == ["test"]


# =============================================================================
# PrefixPathSpec Tests
# =============================================================================


def test_prefix_schema_basic():
    """Test basic prefix_schema with string args and separator."""
    spec = prefix_schema("arg1", "arg2", "/", "arg3")
    parts = spec.build(
        {"arg1": "value1", "arg2": "value2", "arg3": "value3"}, "my_func"
    )

    # arg1 and arg2 grouped together, then separator, then arg3
    assert len(parts) == 3
    assert parts[0] == {"arg1": "value1", "arg2": "value2"}
    assert parts[1] == {"arg3": "value3"}
    assert parts[2] == "my_func"


def test_prefix_schema_with_remaining():
    """Test that unused args are added as suffix."""
    spec = prefix_schema("arg1", "/", "arg2")
    parts = spec.build(
        {"arg1": "value1", "arg2": "value2", "arg3": "value3", "arg4": "value4"},
        "my_func",
    )

    assert parts[0] == {"arg1": "value1"}
    assert parts[1] == {"arg2": "value2"}
    assert parts[2] == "my_func"
    assert parts[3] == {"arg3": "value3", "arg4": "value4"}


def test_prefix_schema_with_tuple():
    """Test prefix_schema with tuple for explicit grouping."""
    spec = prefix_schema(("arg1", "arg2"), "/", ("arg3",))
    parts = spec.build(
        {"arg1": "value1", "arg2": "value2", "arg3": "value3"}, "my_func"
    )

    assert len(parts) == 3
    assert parts[0] == {"arg1": "value1", "arg2": "value2"}
    assert parts[1] == {"arg3": "value3"}
    assert parts[2] == "my_func"


def test_prefix_schema_with_set():
    """Test prefix_schema with set for explicit grouping."""
    spec = prefix_schema({"arg1"}, "/", {"arg2"})
    parts = spec.build({"arg1": "value1", "arg2": "value2"}, "my_func")

    assert len(parts) == 3
    assert parts[0] == {"arg1": "value1"}
    assert parts[1] == {"arg2": "value2"}
    assert parts[2] == "my_func"


def test_prefix_schema_with_list():
    """Test prefix_schema with list for value-only paths (no key names)."""
    spec = prefix_schema(["arg1", "arg2"], "/", "arg3")
    parts = spec.build(
        {"arg1": "value1", "arg2": "value2", "arg3": "value3"}, "my_func"
    )

    # List creates a list of values (no keys)
    assert parts[0] == ["value1", "value2"]
    assert parts[1] == {"arg3": "value3"}
    assert parts[2] == "my_func"


def test_prefix_schema_mixed():
    """Test prefix_schema with mixed arg types."""
    spec = prefix_schema("dataset", "split", "/", ["model_type"], "/", ("epochs",))
    parts = spec.build(
        {
            "dataset": "mnist",
            "split": "train",
            "model_type": "cnn",
            "epochs": 100,
            "lr": 0.01,
        },
        "train_fn",
    )

    assert parts[0] == {"dataset": "mnist", "split": "train"}
    assert parts[1] == ["cnn"]
    assert parts[2] == {"epochs": 100}
    assert parts[3] == "train_fn"
    assert parts[4] == {"lr": 0.01}


def test_prefix_schema_no_remaining():
    """Test prefix_schema when all args are used."""
    spec = prefix_schema("arg1", "/", "arg2")
    parts = spec.build({"arg1": "value1", "arg2": "value2"}, "my_func")

    # No suffix dict when all args used
    assert len(parts) == 3
    assert parts[0] == {"arg1": "value1"}
    assert parts[1] == {"arg2": "value2"}
    assert parts[2] == "my_func"


def test_prefix_schema_include_remaining_false():
    """Test prefix_schema with include_remaining=False."""
    spec = prefix_schema("arg1", "/", "arg2", include_remaining=False)
    parts = spec.build(
        {"arg1": "value1", "arg2": "value2", "arg3": "value3", "arg4": "value4"},
        "my_func",
    )

    # Should NOT include arg3 and arg4 in suffix
    assert len(parts) == 3
    assert parts[0] == {"arg1": "value1"}
    assert parts[1] == {"arg2": "value2"}
    assert parts[2] == "my_func"


def test_prefix_schema_with_object_identity():
    """Test that prefix_schema resolves object identities."""
    registry = ObjectIdentityRegistry()
    obj = _TestObject({"complex": "data"})
    registry.identify(obj, "simple_name")

    spec = prefix_schema("model", "/", "dataset")
    parts = spec.build({"model": obj, "dataset": "mnist"}, "my_func", registry)

    assert parts[0] == {"model": "simple_name"}
    assert parts[1] == {"dataset": "mnist"}


def test_prefix_schema_invalid_arg():
    """Test that invalid arg types raise ValueError."""
    spec = prefix_schema("arg1", 123)  # 123 is invalid

    with pytest.raises(ValueError, match="Invalid prefix arg"):
        spec.build({"arg1": "value1"}, "my_func")


def test_storage_cache_with_prefix_schema(fs: fake_filesystem.FakeFilesystem):
    """Test @storage.cache with prefix_schema."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    counter = 0

    @storage.cache(prefix_schema("dataset", "/", "model"), fmt="json")
    def train(dataset, model, epochs=10):
        nonlocal counter
        counter += 1
        return {"result": f"{dataset}_{model}_{epochs}"}

    assert counter == 0

    # First call computes
    result = train("mnist", "cnn", epochs=20)
    assert result == {"result": "mnist_cnn_20"}
    assert counter == 1

    # Second call uses cache
    cached_result = train("mnist", "cnn", epochs=20)
    assert cached_result == result
    assert counter == 1

    # Different epochs should cache separately (epochs is in remaining args)
    result2 = train("mnist", "cnn", epochs=30)
    assert result2 == {"result": "mnist_cnn_30"}
    assert counter == 2

    # Check that path contains key:value format
    path = train.get_prefix_path("mnist", "cnn", epochs=20)
    path_str = str(path)
    assert "dataset:mnist" in path_str
    assert "model:cnn" in path_str


# =============================================================================
# Convenience Function Tests
# =============================================================================


def test_load_from_path(fs: fake_filesystem.FakeFilesystem):
    """Test load_from_path convenience function."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    test_obj = {"key": "value", "number": 42}

    # Save some data
    path, _ = storage.save(test_obj, "experiments", {"model": "cnn"})

    # Load using the convenience function (path.parent is the storage dir)
    loaded = load_from_path(path.parent)
    assert loaded == test_obj


def test_load_metadata_from_path(fs: fake_filesystem.FakeFilesystem):
    """Test load_metadata_from_path convenience function."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)
    test_obj = {"key": "value"}

    # Save some data
    path, _ = storage.save(test_obj, "experiments", {"model": "cnn"})

    # Load metadata using the convenience function
    metadata = load_metadata_from_path(path.parent)
    assert metadata.timestamp is not None
    assert metadata.path_parts == ["experiments", {"model": "cnn"}]


def test_load_from_path_with_version(fs: fake_filesystem.FakeFilesystem):
    """Test load_from_path with versioned storage."""
    root = "/tmp/storage_test"
    fs.create_dir(root)

    storage = Storage(root, verbose=False)

    # Save with versioning
    path1, _ = storage.save({"v": 1}, "test", version=True)
    path2, _ = storage.save({"v": 2}, "test", version=True)

    # Load specific versions using their paths
    loaded1 = load_from_path(path1.parent)
    loaded2 = load_from_path(path2.parent)

    assert loaded1 == {"v": 1}
    assert loaded2 == {"v": 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
