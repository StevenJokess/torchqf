import torch
import torch.functional as fn

from .tensor import steps


def compound(input: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    """
    Returns the product of all elements in the `input` tensor.

        Compound[r] = [ \prod_i (1 + r[i]) ] - 1

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int | tuple[int], optional
        The dimension or dimensions to reduce.
    keepdim : bool, default=False
        Whether the output tensor has dim retained or not.

    Examples
    --------
    >>> import torchqf

    >>> input = torch.linspace(0.01, 0.06, 6).reshape(2, 3)
    >>> torchqf.compound(input)
    tensor(0.2283)
    >>> torchqf.compound(input, dim=-1)
    tensor([0.0611, 0.1575])
    """
    if dim is None:
        return torch.prod(1.0 + input) - 1.0
    else:
        return torch.prod(1.0 + input, dim=dim, keepdim=keepdim) - 1.0


def cumcompound(input, dim):
    """
    Returns the cumulative compounding of elements of `input` in the dimension dim.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int
        The dimension to do the operation over.

    Examples
    --------
    >>> import torchqf

    >>> input = torch.linspace(0.01, 0.06, 6).reshape(2, 3)
    >>> torchqf.cumcompound(input, dim=-1)
    tensor([[0.0100, 0.0302, 0.0611],
            [0.0400, 0.0920, 0.1575]])
    """
    return torch.cumprod(1.0 + input, dim) - 1.0


def pv(input: torch.Tensor, time, rate=0.0) -> torch.Tensor:
    """
    Return the present value of a cashflow stream (`input`).

        PV[t] = cashflow[t] * exp(- r[t] * t)

    Parameters
    ----------
    input : Tensor
        Cashflow stream.
    time : float | Tensor
        Time(s) at which cashflows accrue.
        If `float`, evenly spaced time-steps until `time`.
    rate : float | Tensor
        Instantaneous rate.

    Returns
    -------
    pv : Tensor
        Present value.

    Shape
    -----
    input : :math:`(*, T)`
        `*` means any number of additional dimensions.
        :math:`T` means the number of time steps.
    time : :math:`(*, T)`
        The same shape with the input.
    output : :math:`(*, T)`
        The same shape with the input.

    Examples
    --------
    >>> import torchqf

    >>> input = torch.tensor([
    ...     [ 1.0, -0.1, -0.1, -0.1],
    ...     [ 2.0, -0.2, -0.2, -0.2]])
    >>> torchqf.pv(input, 4.0, 0.01)
    tensor([[ 0.9900, -0.0980, -0.0970, -0.0961],
            [ 1.9801, -0.1960, -0.1941, -0.1922]])
    """
    if isinstance(time, (int, float)):
        n_steps = input.size(-1)
        time = steps(time, steps=n_steps)
    if isinstance(rate, float):
        rate = torch.tensor(rate)

    return input * torch.exp(-rate * time)


def npv(input, time, rate=0.0, keepdim=False) -> torch.Tensor:
    """
    Returns the net present value of a cash flowsteam (`input`).

        NPV = \sum_t cashflow[t] * exp(- r[t] * t)

    Parameters
    ----------
    input : Tensor
        Cashflow stream.
    time : float | Tensor
        Time(s) at which cashflows accrue.
        If `float`, evenly spaced time-steps until `time`.
    rate : float | Tensor
        Instantaneous rate.
    keepdim : bool, default=False
        Whether the output tensor has dim retained or not.

    Shape
    -----
    input : :math:`(*, T)`
        `*` means any number of additional dimensions.
        :math:`T` means the number of time steps.
    time : :math:`(*, T)`
        The same shape with the input.
    output : :math:`(*)`
        If `keepdims=True`, :math:`(*, T)`.

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
    return pv(input, time, rate).sum(-1, keepdim=keepdim)


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
