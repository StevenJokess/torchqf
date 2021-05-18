import torch


def steps(end: float, steps=None, dtype=None, device=None) -> torch.Tensor:
    """
    Create a one-dimensional tensor of size `steps` whose values are evenly spaced
    zero to `end`, exclusive of zero. That is, the value are:

        `steps(end, steps) = torch.linspace(0, end, steps + 1)[1:]`

    Parameters
    ----------
    end : float
    steps : int, default None

    Examples
    --------
    >>> import torchqf
    >>> torchqf.steps(1.0, steps=5)
    tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
    """
    return torch.linspace(0.0, end, steps + 1, dtype=dtype, device=device)[1:]
