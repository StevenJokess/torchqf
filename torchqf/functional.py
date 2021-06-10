import torch
import torch.functional as fn
from torch import Tensor

from .tensor import steps


def simple_to_compound(rate: Tensor, time) -> Tensor:
    """Converts simple rate into compound rate."""


def compound(input: Tensor, dim: int = None, keepdim: bool = False) -> Tensor:
    """Returns the product of all elements in the `input` tensor.

    .. math::

        \\mathrm{out} = [ \\prod_i (1 + \\mathrm{input}[i]) ] - 1


    Args:
        input (`torch.Tensor`): The input tensor.
        dim (int or tuple[int], optional): The dimension or dimensions to reduce.
        keepdim (bool, default=False): Whether the output tensor has dim retained or not.

    Examples:

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


def cumcompound(input: Tensor, dim: int) -> Tensor:
    """Returns the cumulative compounding of elements of `input` in the dimension dim.

    Args:
        input (`torch.Tensor`): The input tensor.
        dim (int): The dimension to do the operation over.

    Examples:

        >>> import torchqf

        >>> input = torch.linspace(0.01, 0.06, 6).reshape(2, 3)
        >>> torchqf.cumcompound(input, dim=-1)
        tensor([[0.0100, 0.0302, 0.0611],
                [0.0400, 0.0920, 0.1575]])
    """
    return torch.cumprod(1.0 + input, dim) - 1.0


def pv(input: Tensor, time, rate: float = 0.0) -> Tensor:
    """Returns the present value of a cashflow stream (`input`).

        PV[t] = cashflow[t] * exp(- r[t] * t)

    Args:
        input (`torch.Tensor`)
            Cashflow stream.
        time (float | `Tensor`): Time(s) at which cashflows accrue.
            If `float`, evenly spaced time-steps until `time`.
        rate (float | `Tensor`): Instantaneous rate.

    Returns:
        (`Tensor`): Present value.

    Shape:

        - input : :math:`(*, T)`
            `*` means any number of additional dimensions.
            :math:`T` means the number of time steps.
        - time : :math:`(*, T)`
            The same shape with the input.
        - output : :math:`(*, T)`
            The same shape with the input.

    Examples:

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


def npv(input: torch.Tensor, time, rate=0.0, keepdim=False) -> torch.Tensor:
    """Returns the net present value of a cash flowsteam (`input`).

        \\mathrm{NPV} = \\sum_t \\mathrm{cashflow}[t] * \\exp(- r[t] * t)

    Args:
        input (`Tensor`): Cashflow stream.
        time (float | Tensor): Time(s) at which cashflows accrue.
            If `float`, evenly spaced time-steps until `time`.
        rate (float | Tensor): Instantaneous rate.
            TODO(simaki): compound rate
        keepdim (bool, default=False): Whether the output tensor has dim
            retained or not.

    Shape:

        - input : :math:`(*, T)`
            `*` means any number of additional dimensions.
            :math:`T` means the number of time steps.
        - time : :math:`(*, T)`
            The same shape with the input.
        - output : :math:`(*)`
            If `keepdims=True`, :math:`(*, 1)`.

    Examples:

        >>> import torchqf

        >>> input = torch.tensor([
        ...     [ 1.0, -0.1, -0.1, -0.1],
        ...     [ 2.0, -0.2, -0.2, -0.2]])
        >>> torchqf.npv(input, 4.0, 0.01)
        tensor([0.6989, 1.3978])

        Net present value of a bond with 1000 face-value,
        semiannualy accrued 2%/year coupon rate, and two-year expiry.
        Discount rate is chosen to 5%.

        >>> input = torch.tensor([10, 10, 10, 1010])
        >>> time = torch.tensor([0.5, 1.0, 1.5, 2.0])
        >>> torchqf.npv(input, time, rate=0.05)
        tensor(942.4286)
    """
    return pv(input, time=time, rate=rate).sum(-1, keepdim=keepdim)


def european_payoff(input, strike=1.0, call=True, keepdim=False):
    """Returns the payoff of a European option

    Args:
        input (Tensor): The price of the underlying asset.
        strike (float): The strike price
        call (bool): Specifies call or put.

    Shape:

        - input : :math:`(*, T)`
        - output : :math:`(*)`

    Returns:
        Tensor: Payoff

    Examples:

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
    """Evaluate log return of prices.

    Args:
        TODO(simaki)

    Shape:
        - input
            :math:`(*, T)`
            T is the number of time steps.
            `*` means any number of additional dimensions.
        - output
            :math:`(*, T - 1)`

    Returns:
        TODO(simaki)

    Examples:

        >>> # TODO(simaki)
    """
    return (input[..., 1:] / input[..., :-1]).log()


def softwhere(input, x=0.0, y=1.0, width=1.0):
    """Soft version of `torch.where(input > 0, x, y)`

    The operation is defined as:
    .. math::

        out = p * x + (1 - p) * y
        p = sigmoid(input / width)

    Args:
        - input : Tensor
        - threshold : float | Tensor
        - x : float | Tensor
        - y : float | Tensor
        - width : float | Tensor

    Returns:
        TODO(simaki)

    Shape:
        - input : :math:`(*)`
    """
    p = fn.sigmoid(input / width)
    return p * x + (1 - p) * y
