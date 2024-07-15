from pyfakefs import fake_filesystem
from blackhc.project.utils import simple_storage
import os
import pytest


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
    
    @simple_storage.cache(prefix_args=["arg1"], root=root, force_format="json")
    def test_function(arg1, arg2):
        return {"result": arg1 + arg2}
            
    # Test loading directly from cache
    with pytest.raises(FileNotFoundError):
        loaded_result = test_function.load(1, 2)
    
    result = test_function(1, 2)
    assert result == {"result": 3}
    
    cached_result = test_function(1, 2)
    assert cached_result == result
    
    # Test loading directly from cache
    loaded_result = test_function.load(1, 2)
    assert loaded_result == result
    
    # Test recomputing the result
    recomputed_result = test_function.recompute(1, 2)
    assert recomputed_result == result
    # We should have two subdirectories now
    assert len(os.listdir(test_function.get_prefix_path(1, 2))) == 2
