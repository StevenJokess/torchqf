import torch

import torchqf


def test_steps():
    result = torchqf.steps(1.0, 5)
    expect = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    assert torch.allclose(result, expect)


def test_npv():
    # not discounted for rate=0.0
    input = torch.tensor([1.0, 2.0]).reshape(-1, 1)
    result = torchqf.npv(input, time=1.0, rate=0.0)
    expect = torch.tensor([1.0, 2.0])
    assert torch.allclose(result, expect)

    # not discounted for time=0.0
    input = torch.tensor([1.0, 2.0]).reshape(-1, 1)
    result = torchqf.npv(input, time=0.0, rate=0.1)
    expect = torch.tensor([1.0, 2.0])
    assert torch.allclose(result, expect)
