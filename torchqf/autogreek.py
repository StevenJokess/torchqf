import torch

from .model._utils import _parse_spot


def delta(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """
    Parameters
    ----------
    pricer : callable
        Pricing formula of a derivative.
    spot : Tensor
        Spot price of underlier
    create_graph : bool, default=False
    **kwargs
        Other parameters passed to pricer.

    Examples
    --------
    >>> pricer = lambda spot, expiry: spot * expiry
    >>> spot = torch.tensor([1.0, 2.0, 3.0])
    >>> expiry = torch.tensor([2.0, 3.0, 4.0])
    >>> delta(pricer, spot=spot, expiry=expiry)
    tensor([2., 3., 4.])
    """
    spot = kwargs.get("spot")
    strike = kwargs.get("strike")
    moneyness = kwargs.get("moneyness")
    log_moneyness = kwargs.get("log_moneyness")

    if log_moneyness is not None or moneyness is not None:
        if strike is None:
            # Since delta does not depend on strike,
            # assign an arbitrary value (1.0) to strike if not given.
            strike = torch.tensor(1.0)
            kwargs["strike"] = strike

    spot = _parse_spot(
        spot=spot, moneyness=moneyness, log_moneyness=log_moneyness, strike=strike
    ).requires_grad_()
    kwargs["spot"] = spot
    if "moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        del kwargs["moneyness"]
    if "log_moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        del kwargs["log_moneyness"]
    price = pricer(**kwargs)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def gamma(pricer, *, create_graph: bool = False, **kwargs) -> torch.Tensor:
    """
    Examples
    --------
    >>> @torch.enable_grad()
    ... def pricer(spot, expiry):
    ...     return spot * (expiry ** 2)
    >>> spot = torch.tensor([1.0, 2.0, 3.0])
    >>> expiry = torch.tensor([2.0, 3.0, 4.0])
    >>> gamma(pricer, spot=spot, expiry=expiry)
    """
    spot = kwargs.get("spot")
    strike = kwargs.get("strike")
    moneyness = kwargs.get("moneyness")
    log_moneyness = kwargs.get("log_moneyness")

    spot = _parse_spot(
        spot=spot, moneyness=moneyness, log_moneyness=log_moneyness, strike=strike
    ).requires_grad_()

    kwargs["spot"] = spot
    if "moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        del kwargs["moneyness"]
    if "log_moneyness" in kwargs:
        # lest moneyness is used to compute price and grad wrt spot cannot be computed
        del kwargs["log_moneyness"]
    tensor_delta = delta(pricer, create_graph=True, **kwargs).requires_grad_()
    return torch.autograd.grad(
        tensor_delta,
        inputs=spot,
        grad_outputs=torch.ones_like(tensor_delta),
        create_graph=create_graph,
        allow_unused=True,
    )[0]
