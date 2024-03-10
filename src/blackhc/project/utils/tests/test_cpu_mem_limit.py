import pytest
import torch
import platform

from blackhc.project.utils import cpu_memory


def is_apple_silicon():
    return platform.machine() == 'arm64'


@pytest.mark.skipif(is_apple_silicon(), reason="Apple Silicon")
@pytest.mark.forked
def test_cpu_mem_limit():
    # 128 MB (128/4M float32)
    tensor = torch.empty((128, 1024, 1024 // 4), dtype=torch.float32)
    tensor.resize_(1)

    cpu_memory.set_cpu_memory_limit(0.25)

    # 512 MB (128/4M float32)
    with pytest.raises(RuntimeError):
        torch.empty((512, 1024, 1024 // 4), dtype=torch.float32)
