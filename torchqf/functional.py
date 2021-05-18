import torch
import torch.functional as fn

from .tensor import steps


def npv(input, time, rate=0.0, keepdim=False) -> torch.Tensor:
    """
    Returns the net present value of a cash flow steam (`input`).

        NPV = \sum_t cashflow[t] * exp(- r[t] * t)

    Parameters
    ----------
    input : Tensor
        Cashflow stream.
    time : Tensor | float, default=1.0
        float :
            split the time until `time`
    rate : Tensor, shape (T,) | float
        instantaneous rate
    keepdim : bool, default False

    Shape
    -----
    input : :math:`(*, T)`
        :math:`T` means the number of time steps.
    time : :math:`(*, T)`
    output : :math:`(*)`
        Net present value of the cashflow stream.

    Examples
    --------
    >>> input = torch.tensor([
    ...     [ 1.0, -0.1, -0.1, -0.1],
    ...     [ 2.0, -0.2, -0.2, -0.2]])
    >>> npv(input, 4.0, 0.01)
    tensor([0.6989, 1.3978])

    This returns the present value for a cashflow with shape `(*, 1)`.

    >>> input = torch.tensor([1.0])
    >>> npv(input, time=0.5, rate=0.01)
    tensor(0.9950)
    """
    if isinstance(time, (int, float)):
        n_steps = input.size(-1)
        time = steps(time, steps=n_steps)
    if isinstance(rate, float):
        rate = torch.tensor(rate)

    return (input * torch.exp(-rate * time)).sum(-1, keepdim=keepdim)


def european_payoff(input, strike=1.0, call=True, keepdim=False):
    """
    Return the payoff of a European option

    Parameters
    ----------
    input : Tensor, size (*, T)
        The last dimension is time
        `*` means any number of additional dimensions.
    strike : float
        The strike price
    call : bool
        Specifies call/put.

    Shape
    -----
    Input : :math:`(*, T)`
    Output : :math:`(*)`

    Returns
    -------
    payoff : Tensor, size (*)

    Examples
    --------
    >>> # TODO
    """
    if call:
        payoff = fn.relu(input[..., -1] - strike)
    else:
        payoff = fn.relu(strike - input[..., -1])

    if keepdim:
        out = torch.zeros_like(input)
        out[..., -1] = payoff
        return out
    else:
        return payoff


def log_return(input):
    """
    Evaluate log return of prices.

    Shape
    -----
    input
        :math:`(*, T)`
        T is the number of time steps.
        `*` means any number of additional dimensions.
    output
        :math:`(*, T - 1)`
    """
    return (input[..., 1:] / input[..., :-1]).log()
