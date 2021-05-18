import torch

from ..tensor import steps


def generate_brownian(
    size, time, drift=0.0, volatility=0.2, init_value=0.0, dtype=None, device=None
):
    """
    Generate time-series that follows Brownian motion.

    Parameters
    ----------
    size : tuple[int]
        The last dimension is time
    time : Tensor, shape (T, *)
        Time steps.
    volatility : float, default 0.2
        Volatility of the process.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> generate_brownian((2, 5), time=0.1)
    tensor([[ 0.0095,  0.0036,  0.0066,  0.0065, -0.0318],
            [ 0.0043,  0.0661, -0.0114,  0.0196, -0.0242]])
    """
    assert dtype is None, "not supported"
    assert device is None, "not supported"

    n_steps = size[-1]

    if not isinstance(time, torch.Tensor):
        time = steps(time, n_steps)

    dt = torch.empty_like(time)
    dt[..., 0] = time[0] - 0.0
    dt[..., 1:] = time[1:] - time[:-1]

    drift_term = drift * time
    random_term = (volatility * torch.randn(size) * dt.sqrt()).cumsum(0)

    return init_value + drift_term + random_term


def generate_geometric_brownian(
    size, time, drift=0.0, volatility=0.2, init_value=1.0, dtype=None, device=None
):
    """
    Generate time-series that follows geometric Brownian motion.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> generate_geometric_brownian((2, 5), time=0.1)
    tensor([[1.0092, 1.0028, 1.0054, 1.0049, 0.9668],
            [1.0039, 1.0675, 0.9875, 1.0181, 0.9741]])
    """
    assert dtype is None, "not supported"
    assert device is None, "not supported"

    drift = drift - (volatility ** 2 / 2)
    brown = generate_brownian(
        size, time, drift=drift, volatility=volatility, dtype=dtype, device=device
    )

    return init_value * torch.exp(brown)
