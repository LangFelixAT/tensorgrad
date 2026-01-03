import numpy as np
import pytest
import importlib

@pytest.fixture
def rng():
    return np.random.default_rng(0)

def _has_cupy():
    return importlib.util.find_spec("cupy") is not None

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not _has_cupy():
        pytest.skip("cupy not installed")
    return request.param