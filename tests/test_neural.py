import pytest
import torch

from electricity_market.neural import ARCHITECTURES


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_neural_shapes_and_backward(name):
    torch.manual_seed(1)
    model = ARCHITECTURES[name]()
    history = torch.randn(3, 168, 12)
    calendar = torch.zeros(3, 6)
    prediction = model(history, calendar)
    assert prediction.shape == (3, 24)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
