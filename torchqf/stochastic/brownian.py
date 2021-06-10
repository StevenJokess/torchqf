import torch
from torch import Tensor

from ..tensor import steps


def generate_brownian(
    size,
    time,
    drift: float = 0.0,
    volatility: float = 0.2,
    init_value: float = 0.0,
    dtype=None,
    device=None,
) -> Tensor:
    """Generates and returns time-series that follows Brownian motion.

    Args:
        size (torch.Size): The shape of the output tensor.
            The last dimension means the number of time steps.
        time (float | Tensor): The total time length (`float`) or time steps (`Tensor`).
        drift (float, default 0.0): The drift of the process.
        volatility (float, default 0.2): The volatility of the process.
        init_value (float, default 0.0)
            Initial value of the process.
        dtype (`torch.dtype`, optional): The desired data type of returned tensor.
            Default: if None, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (`torch.device`, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        Tensor: The time-series.

    Size:

        - time: :math:`(T,)`
            :math:`T` means the number of time steps.
        - output: :math:`(*, T)`
            The shape specified by `size`.

    Examples:

        >>> from torchqf.stochastic import generate_brownian

        >>> _ = torch.manual_seed(42)
        >>> generate_brownian((2, 5), time=0.1)
        tensor([[ 0.0095,  0.0132,  0.0198,  0.0263, -0.0054],
                [-0.0053,  0.0572,  0.0391,  0.0522,  0.0598]])
    """
    assert dtype is None, "not supported"
    assert device is None, "not supported"

    n_steps = size[-1]

    if not isinstance(time, torch.Tensor):
        time = steps(time, n_steps)  # shape : (T,)

    dt = torch.empty_like(time)
    dt[0] = time[0] - 0.0
    dt[1:] = time[1:] - time[:-1]

    drift_term = drift * time
    random_term = (volatility * torch.randn(size) * dt.sqrt()).cumsum(-1)

    return init_value + drift_term + random_term


def generate_geometric_brownian(
    size,
    time,
    drift: float = 0.0,
    volatility: float = 0.2,
    init_value: float = 1.0,
    dtype=None,
    device=None,
) -> Tensor:
    """Generates and returns time-series that follows geometric Brownian motion.

    Args:
        size (tuple[int]): The shape of the output tensor.
            The last dimension means the number of time steps.
        time (float | Tensor): The total time length (`float`) or time steps (`Tensor`).
        drift (float, default 0.0): The drift of the process.
        volatility (float, default 0.2): The volatility of the process.
        init_value (float, default 0.0): Initial value of the process.
        dtype (`torch.dtype`, optional): The desired data type of returned tensor.
            Default: if None, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (`torch.device`, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
    -------
        Tensor: The time-series.

    Size:

        - time: :math:`(T,)`
            :math:`T` means the number of time steps.
        - output: :math:`(*, T)`
            The shape specified by `size`.

    Examples:

        >>> from torchqf.stochastic import generate_geometric_brownian

        >>> _ = torch.manual_seed(42)
        >>> generate_geometric_brownian((2, 5), time=0.1)
        tensor([[1.0092, 1.0124, 1.0188, 1.0250, 0.9926],
                [0.9943, 1.0580, 1.0387, 1.0519, 1.0595]])
    """
    assert dtype is None, "not supported"
    assert device is None, "not supported"

    brown = generate_brownian(
        size,
        time=time,
        drift=drift - volatility ** 2 / 2,
        volatility=volatility,
        dtype=dtype,
        device=device,
    )

    return init_value * torch.exp(brown)
