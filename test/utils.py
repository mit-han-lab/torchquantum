
import numpy as np
import torch

def check_all_close(a, b, rtol=1e-5, atol=1e-4):
    """Check that all elements of a and b are close."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=rtol, atol=atol)
