from pyfakefs import fake_filesystem
from blackhc.project.utils import simple_storage
import os
import pytest
import enum


def test_arg_to_path_fragment():
    assert simple_storage.value_to_path_fragment(123) == "123"
    assert simple_storage.value_to_path_fragment(123.456) == "123.456"
    assert simple_storage.value_to_path_fragment("test_string") == "test_string"
    assert simple_storage.value_to_path_fragment([1, 2, 3]) == "1+2+3"
    assert simple_storage.value_to_path_fragment((1, 2, 3)) == "(1,2,3)"
    assert simple_storage.value_to_path_fragment({"key": "value"}) == "{key:value}"
    assert simple_storage.value_to_path_fragment({None: "value"}) == "{None:value}"
    assert simple_storage.value_to_path_fragment(123.0) == "123"
    assert simple_storage.value_to_path_fragment(["a", "b", "c"]) == "a+b+c"
    assert simple_storage.value_to_path_fragment({"a": 1, "b": 2}) == "{a:1,b:2}"

    class TestEnum(enum.Enum):
        OPTION_A = "OptionA"
        OPTION_B = "OptionB"

    assert simple_storage.value_to_path_fragment(TestEnum.OPTION_A) == "OptionA"
    assert simple_storage.value_to_path_fragment(TestEnum.OPTION_B) == "OptionB"
    
def test_arg_to_path_fragment_with_int_enum():
    class IntEnum(enum.Enum):
        OPTION_1 = 1
        OPTION_2 = 2

    assert simple_storage.value_to_path_fragment(IntEnum.OPTION_1) == "OPTION_1"
    assert simple_storage.value_to_path_fragment(IntEnum.OPTION_2) == "OPTION_2"
    
    
def test_dict_to_path_fragment():
    assert simple_storage.dict_to_path_fragment({"key": "value"}) == "key:value"
    assert simple_storage.dict_to_path_fragment({"key1": "value1", "key2": "value2"}) == "key1:value1_key2:value2"
    assert simple_storage.dict_to_path_fragment({"key": 123}) == "key:123"
    assert simple_storage.dict_to_path_fragment({"key": 123.456}) == "key:123.456"
    assert simple_storage.dict_to_path_fragment({"key": [1, 2, 3]}) == "key:1+2+3"
    assert simple_storage.dict_to_path_fragment({"key": (1, 2, 3)}) == "key:(1,2,3)"
    assert simple_storage.dict_to_path_fragment({"key": {"subkey": "subvalue"}}) == "key:{subkey:subvalue}"
    assert simple_storage.dict_to_path_fragment({"key": None}) == "key:None"
    assert simple_storage.dict_to_path_fragment({"key": 123.0}) == "key:123"
    assert simple_storage.dict_to_path_fragment({"key1": "value1", "key2": None}) == "key1:value1_key2:None"
    assert simple_storage.dict_to_path_fragment({None: 123.0}) == "123"


def test_save_and_load_pkl(fs: fake_filesystem.FakeFilesystem):
    test_obj = {"key": "value"}
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    path = simple_storage.save_pkl(test_obj, "test", root=root)
    loaded_obj = simple_storage.load_pkl("test", root=root)
    
    assert test_obj == loaded_obj
    assert os.path.exists(f"{path}.meta.json")


def test_save_and_load_json(fs: fake_filesystem.FakeFilesystem):
    test_obj = {"key": "value"}
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    path = simple_storage.save_json(test_obj, "test", root=root)
    loaded_obj = simple_storage.load_json("test", root=root)
    
    assert test_obj == loaded_obj
    assert os.path.exists(f"{path}.meta.json")


def test_save_and_load_pt(fs: fake_filesystem.FakeFilesystem):
    if simple_storage.torch is None:
        pytest.skip("torch is not available")
    
    test_obj = simple_storage.torch.tensor([1, 2, 3])
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    path = simple_storage.save_pt(test_obj, "test", root=root)
    loaded_obj = simple_storage.load_pt("test", root=root)
    
    assert simple_storage.torch.equal(test_obj, loaded_obj)
    assert os.path.exists(f"{path}.meta.json")


def test_save_pkl_or_json(fs: fake_filesystem.FakeFilesystem):
    test_obj = {"key": "value"}
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    path = simple_storage.save_pkl_or_json(test_obj, "test", root=root)
    loaded_obj = simple_storage.load("test", root=root)
    
    assert test_obj == loaded_obj
    assert os.path.exists(f"{path}.meta.json")


def test_collect_metadata(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    metadata = simple_storage.collect_metadata("test")
    assert "timestamp" in metadata
    assert "git" in metadata
    assert "wandb" in metadata
    assert "parts" in metadata


def test_cache_decorator(fs: fake_filesystem.FakeFilesystem):
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    counter = 0
    
    @simple_storage.cache(path_schema=simple_storage.prefix_schema(["arg1"]), root=root, force_format="json")
    def test_function(arg1, arg2):
        nonlocal counter
        counter += 1
        return {"result": arg1 + arg2}
            
    # Test loading directly from cache
    with pytest.raises(FileNotFoundError):
        loaded_result = test_function.load(1, 2)
        
    assert counter == 0
    
    result = test_function(1, 2)
    assert result == {"result": 3}
    assert counter == 1
    
    cached_result = test_function(1, 2)
    assert cached_result == result
    assert counter == 1
    
    # Test loading directly from cache
    loaded_result = test_function.load(1, 2)
    assert loaded_result == result
    assert counter == 1
    
    # Test recomputing the result
    recomputed_result = test_function.recompute(1, 2)
    assert recomputed_result == result
    assert counter == 2
    
    # We should have two subdirectories now
    assert len(os.listdir(test_function.get_prefix_path(1, 2))) == 2


def test_save_and_load_with_timestamp(fs: fake_filesystem.FakeFilesystem):
    test_obj = {"key": "value"}
    root = "/tmp/blackhc.project"
    fs.CreateDirectory(root)
    
    # Test saving with NOW timestamp
    path_now = simple_storage.save_json(test_obj, "test", root=root, timestamp=simple_storage.Timestamp.NOW)
    loaded_obj_now = simple_storage.load_json("test", root=root, timestamp=simple_storage.Timestamp.LATEST)
    assert test_obj == loaded_obj_now
    assert os.path.exists(f"{path_now}meta.json")
    
    # Test saving with a specific timestamp
    specific_timestamp = "2023-01-01T00:00:00"
    path_specific = simple_storage.save_json(test_obj, "test2", root=root, timestamp=specific_timestamp)
    loaded_obj_specific = simple_storage.load_json("test2", root=root, timestamp=specific_timestamp)
    assert test_obj == loaded_obj_specific
    assert os.path.exists(f"{path_specific}meta.json")
    
    specific_timestamp = "2024-01-01T00:00:00"
    path_specific = simple_storage.save_json(test_obj, "test2", root=root, timestamp=specific_timestamp)
    loaded_obj_specific = simple_storage.load_json("test2", root=root, timestamp=specific_timestamp)
    assert test_obj == loaded_obj_specific
    assert os.path.exists(f"{path_specific}meta.json")
    
    # Test loading with LATEST timestamp
    loaded_obj_latest = simple_storage.load_json("test2", root=root, timestamp=simple_storage.Timestamp.LATEST)
    assert test_obj == loaded_obj_latest
    
    
def test_part_schema():
    # Test with a simple schema
    schema = simple_storage.part_schema(
        simple_storage.PartLiteral("literal"),
        simple_storage.PartKW("key", "value"),
        simple_storage.PartIdentifier,
        ("arg1", simple_storage.PartLiteral("nested_literal"))
    )

    bound_arguments = {"arg1": "arg1_value"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        "literal",
        {"key": "value"},
        "test_id",
        {"arg1": "arg1_value", None: "nested_literal"}
    ]

    # Test with a more complex schema
    schema = simple_storage.part_schema(
        simple_storage.PartLiteral("fixed"),
        simple_storage.PartKW("param", "test"),
        simple_storage.PartIdentifier,
        ("arg1", "arg2", simple_storage.PartKW("a", "end_literal"))
    )

    bound_arguments = {"arg1": "value1", "arg2": "value2"}
    identifier = "complex_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        "fixed",
        {"param": "test"},
        "complex_id",
        {"arg1": "value1", "arg2": "value2", "a": "end_literal"}
    ]

    # Test with missing argument
    schema = simple_storage.part_schema(
        simple_storage.PartLiteral("start"),
        simple_storage.PartIdentifier,
        ("arg1", simple_storage.PartLiteral("middle"), "arg2")
    )

    bound_arguments = {"arg1": "value1"}
    identifier = "missing_arg_id"
    with pytest.raises(ValueError, match='Missing schema arg'):
        parts = schema(bound_arguments, identifier)

    # Test with nested schema
    schema = simple_storage.part_schema(
        simple_storage.PartLiteral("nested"),
        simple_storage.PartIdentifier,
        ("arg1", simple_storage.PartLiteral("inner_end"))
    )

    bound_arguments = {"arg1": "inner_value"}
    identifier = "nested_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        "nested",
        "nested_id",
        {"arg1": "inner_value", None: "inner_end"}
    ]
    

def test_prefix_schema():
    # Test with simple prefix schema
    schema = simple_storage.prefix_schema("prefix1", "prefix2")

    bound_arguments = {"prefix1": "value1", "prefix2": "value2", "arg1": "arg1_value"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        {"prefix1": "value1", "prefix2": "value2"},
        "test_id",
        {"arg1": "arg1_value"}
    ]

    # Test with prefix schema containing a separator
    schema = simple_storage.prefix_schema("prefix1", "/", "prefix2")

    bound_arguments = {"prefix1": "value1", "prefix2": "value2", "arg1": "arg1_value"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        {"prefix1": "value1"},
        {"prefix2": "value2"},
        "test_id",
        {"arg1": "arg1_value"}
    ]

    # Test with prefix schema containing a set
    schema = simple_storage.prefix_schema({"prefix1", "prefix2"})

    bound_arguments = {"prefix1": "value1", "prefix2": "value2", "arg1": "arg1_value"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        {"prefix1": "value1", "prefix2": "value2"},
        "test_id",
        {"arg1": "arg1_value"}
    ]

    # Test with prefix schema containing a list
    schema = simple_storage.prefix_schema(["prefix1", "prefix2"])

    bound_arguments = {"prefix1": "value1", "prefix2": "value2", "arg1": "arg1_value"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == [
        ["value1", "value2"],
        "test_id",
        {"arg1": "arg1_value"}
    ]
    

def test_template_schema():
    # Test with a simple template schema
    schema = simple_storage.template_schema("prefix/{prefix1}/{prefix2}/{identifier}/suffix")

    bound_arguments = {"prefix1": "value1", "prefix2": "value2"}
    identifier = "test_id"
    parts = schema(bound_arguments, identifier)
    assert parts == ["prefix", "value1", "value2", "test_id", "suffix"]

    # Test with a template schema containing only identifier
    schema = simple_storage.template_schema("{identifier}")

    bound_arguments = {}
    identifier = "only_id"
    parts = schema(bound_arguments, identifier)
    assert parts == ["only_id"]

    # Test with a template schema containing mixed arguments
    schema = simple_storage.template_schema("start/{prefix1}/{identifier}/end")

    bound_arguments = {"prefix1": "value1"}
    identifier = "mixed_id"
    parts = schema(bound_arguments, identifier)
    assert parts == ["start", "value1", "mixed_id", "end"]

    # Test with a template schema containing special characters
    schema = simple_storage.template_schema("{{hello}}/{prefix1}/{identifier}/chars")

    bound_arguments = {"prefix1": "value1"}
    identifier = "special_id"
    parts = schema(bound_arguments, identifier)
    assert parts == ["{hello}", "value1", "special_id", "chars"]

    # Test with a template schema containing missing argument
    schema = simple_storage.template_schema("missing/{prefix1}/{prefix2}/{identifier}")

    bound_arguments = {"prefix1": "value1"}
    identifier = "missing_id"
    try:
        parts = schema(bound_arguments, identifier)
    except KeyError as e:
        assert 'prefix2' in str(e)

    # Test with a final /
    schema = simple_storage.template_schema("final/{prefix1}/{identifier}/")

    bound_arguments = {"prefix1": "value1"}
    identifier = "final_id"
    parts = schema(bound_arguments, identifier)
    assert parts == ["final", "value1", "final_id", ""]


if __name__ == "__main__":
    pytest.main()