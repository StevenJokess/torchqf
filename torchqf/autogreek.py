import torch

from .model._utils import _parse_spot


def delta(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """Computes and returns the delta of a derivative.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool): If True, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
            Default: False.
        **kwargs: Other parameters passed to `pricer`.

    Returns:
        torch.Tensor: The greek of a derivative.

    Examples:

        >>> pricer = lambda spot, expiry: spot * expiry
        >>> spot = torch.tensor([1.0, 2.0, 3.0])
        >>> expiry = torch.tensor([2.0, 3.0, 4.0])
        >>> delta(pricer, spot=spot, expiry=expiry)
        tensor([2., 3., 4.])
    """
    if kwargs.get("strike") is None and kwargs.get("spot") is None:
        # Since delta does not depend on strike,
        # assign an arbitrary value (1.0) to strike if not given.
        kwargs["strike"] = torch.tensor(1.0)

    spot = _parse_spot(**kwargs).requires_grad_()
    kwargs["spot"] = spot

    if "moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        kwargs["moneyness"] = None
    if "log_moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        kwargs["log_moneyness"] = None

    price = pricer(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def gamma(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """Computes and returns the gamma of a derivative.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool): If True, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
            Default: False.
        **kwargs: Other parameters passed to `pricer`.

    Returns:
        torch.Tensor: The greek of a derivative.

    Examples:

        >>> import torchqf

        >>> pricer = lambda spot, expiry: (spot ** 2) * expiry
        >>> spot = torch.tensor([1.0, 2.0, 3.0])
        >>> expiry = torch.tensor([2.0, 3.0, 4.0])
        >>> torchqf.autogreek.gamma(pricer, spot=spot, expiry=expiry)
        tensor([4., 6., 8.])
    """
    spot = _parse_spot(**kwargs).requires_grad_()
    kwargs["spot"] = spot

    if "moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        kwargs["moneyness"] = None
    if "log_moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        kwargs["log_moneyness"] = None

    tensor_delta = delta(pricer, create_graph=True, **kwargs).requires_grad_()
    return torch.autograd.grad(
        tensor_delta,
        inputs=spot,
        grad_outputs=torch.ones_like(tensor_delta),
        create_graph=create_graph,
    )[0]


def theta(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """Computes and returns the theta of a derivative.

    Note:
        theta here is defined as the negative of the price
        differentiated by time remaining to expiry.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool): If True, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
            Default: False.
        **kwargs: Other parameters passed to `pricer`.

    Returns:
        torch.Tensor: The greek of a derivative.

    Examples:

        >>> import torchqf

        >>> pricer = lambda spot, expiry: spot * expiry
        >>> spot = torch.tensor([1.0, 2.0, 3.0])
        >>> expiry = torch.tensor([2.0, 3.0, 4.0])
        >>> torchqf.autogreek.theta(pricer, spot=spot, expiry=expiry)
        tensor([-1., -2., -3.])
    """
    if not isinstance(kwargs.get("expiry"), torch.Tensor):
        raise ValueError

    expiry = kwargs["expiry"].requires_grad_()

    price = pricer(**kwargs)
    # Negative because
    return -torch.autograd.grad(
        price,
        inputs=expiry,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def vega(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """Computes and returns the vega of a derivative.

    Args:
        pricer (callable): Pricing formula of a derivative.
        create_graph (bool): If True, graph of the derivative will be
            constructed, allowing to compute higher order derivative products.
            Default: False.
        **kwargs: Other parameters passed to `pricer`.

    Examples:

        >>> import torchqf

        >>> pricer = lambda spot, volatility: spot * volatility
        >>> spot = torch.tensor([1.0, 2.0, 3.0])
        >>> volatility = torch.tensor([2.0, 3.0, 4.0])
        >>> torchqf.autogreek.vega(pricer, spot=spot, volatility=volatility)
        tensor([1., 2., 3.])
    """
    if not isinstance(kwargs.get("volatility"), torch.Tensor):
        raise ValueError

    volatility = kwargs["volatility"].requires_grad_()

    price = pricer(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=volatility,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]
