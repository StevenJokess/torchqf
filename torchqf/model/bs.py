from cmath import exp
from cmath import log
from logging import root
from random import vonmisesvariate

import torch
from torch.distributions.normal import Normal

from ._utils import _parse_log_moneyness
from ._utils import _parse_spot
from ._utils import root_bisect


class BlackScholesMixin:
    """
    A mixin class to provide useful equations for Black Scholes
    """

    @property
    def N(self):
        """
        normal dist with
        """
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        pdf = lambda input: normal.log_prob(input).exp()
        setattr(normal, "pdf", pdf)
        return normal

    @staticmethod
    def d1(
        volatility: torch.Tensor, log_moneyness: torch.Tensor, expiry: torch.Tensor
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        log_moneyness : Tensor
            ...
        expiry : Tensor
            ...
        volatility : Tensor
            ...

        Returns
        -------
        d1 : Tensor

        Shapes
        ------
        ...
        """
        # TODO(simaki) Use _parse_log_moneyness so users can use spot, strike
        v, s, t = map(torch.as_tensor, (volatility, log_moneyness, expiry))
        return (s + (v ** 2 / 2) * t) / (v * torch.sqrt(t))

    @staticmethod
    def d2(
        volatility: torch.Tensor, log_moneyness: torch.Tensor, expiry: torch.Tensor
    ) -> torch.Tensor:
        # TODO(simaki) Use _parse_log_moneyness so users can use spot, strike
        v, s, t = map(torch.as_tensor, (volatility, log_moneyness, expiry))
        return (s - (v ** 2 / 2) * t) / (v * torch.sqrt(t))


class BSEuropeanOption(BlackScholesMixin):
    """
    Black Scholes formulas for a European option.

    Parameters
    ----------
    is_call : bool, default=True
    """

    def __init__(self, is_call=True):
        self.is_call = is_call

    def price(
        self,
        *,
        volatility: torch.Tensor,
        expiry: torch.Tensor,
        strike: torch.Tensor,
        log_moneyness: torch.Tensor = None,
        moneyness: torch.Tensor = None,
        spot: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns Black-Scholes price of European option.

        Examples
        --------
        >>> volatility = torch.tensor([0.18, 0.20, 0.22])
        >>> moneyness = torch.ones(3)
        >>> expiry = torch.ones(3)
        >>> BSEuropeanOption().price(
        ...     volatility=volatility, moneyness=moneyness, expiry=expiry, strike=1.0)
        tensor([0.0717, 0.0797, 0.0876])
        """
        # TODO(simaki) Use _parse_log_moneyness so users can use spot, strike

        log_moneyness = _parse_log_moneyness(
            spot=spot, strike=strike, moneyness=moneyness, log_moneyness=log_moneyness
        )

        s, t, v = map(torch.as_tensor, (log_moneyness, expiry, volatility))

        n1 = self.N.cdf(self.d1(v, s, t))
        n2 = self.N.cdf(self.d2(v, s, t))

        price = strike * (torch.exp(s) * n1 - n2)

        if not self.is_call:
            price += strike * (torch.exp(s) - 1)  # put-call parity

        return price

    def iv(
        self,
        *,
        price: torch.Tensor,
        expiry: torch.Tensor,
        strike: torch.Tensor,
        log_moneyness: torch.Tensor = None,
        moneyness: torch.Tensor = None,
        spot: torch.Tensor = None,
        differentiable: bool = False,
        precision: float = 1e-6,  # TODO(simaki) determine sensible default for precision
        bisect_lower: float = 0.001,
        bisect_upper: float = 1.0,
        max_iter: int = 100,
    ) -> torch.Tensor:
        """
        Returns implied volatility.

        Parameters
        ----------
        ...
        differentiable : bool, default=True
            Find root with differentiable operation
        bisect_lower : float, default=0.001
        bisect_upper : float, default=0.001

        Examples
        --------
        >>> price = torch.tensor([0.07, 0.08, 0.09])
        >>> moneyness = torch.ones(3)
        >>> expiry = torch.ones(3)
        >>> formula = BSEuropeanOption()
        >>> iv = formula.iv(
        ...     price=price, moneyness=moneyness, expiry=expiry, strike=1.0)
        >>> iv
        tensor([0.1757, 0.2009, 0.2261])
        >>> BSEuropeanOption().price(
        ...     volatility=iv,
        ...     moneyness=moneyness,
        ...     expiry=expiry,
        ...     strike=1.0,
        ... )
        tensor([0.0700, 0.0800, 0.0900])
        """
        assert not differentiable, "not supported"

        log_moneyness = _parse_log_moneyness(
            spot=spot, strike=strike, moneyness=moneyness, log_moneyness=log_moneyness
        )

        get_price = lambda volatility: self.price(
            volatility=volatility,
            log_moneyness=log_moneyness,
            expiry=expiry,
            strike=strike,
        )
        return root_bisect(
            get_price,
            price,
            lower=bisect_lower,
            upper=bisect_upper,
            precision=precision,
            max_iter=max_iter,
        )

    @torch.enable_grad()
    def delta(
        self,
        *,
        volatility: torch.Tensor,
        expiry: torch.Tensor,
        log_moneyness: torch.Tensor = None,
        moneyness: torch.Tensor = None,
        spot: torch.Tensor = None,
        strike: torch.Tensor = None,
        create_graph=False,
    ) -> torch.Tensor:
        """
        Returns Black-Scholes delta of the derivative.

        Examples
        --------
        >>> volatility = torch.tensor([0.18, 0.20, 0.22])
        >>> moneyness = torch.ones(3)
        >>> expiry = torch.ones(3)
        >>> BSEuropeanOption().delta(
        ...     volatility=volatility, moneyness=moneyness, expiry=expiry, strike=1.0)
        tensor([0.5359, 0.5398, 0.5438])
        """
        if log_moneyness is not None or moneyness is not None:
            if strike is not None:
                # Since delta does not depend on strike,
                # assign an arbitrary value (1.0) to strike if not given.
                strike = torch.tensor(1.0)

        spot = _parse_spot(
            spot=spot, strike=strike, moneyness=moneyness, log_moneyness=log_moneyness
        ).requires_grad_()
        price = self.price(
            volatility=volatility, spot=spot, strike=strike, expiry=expiry
        )
        return torch.autograd.grad(
            price,
            inputs=spot,
            grad_outputs=torch.ones_like(price),
            create_graph=create_graph,
        )[0]

    @torch.enable_grad()
    def gamma(
        self,
        *,
        volatility: torch.Tensor,
        expiry: torch.Tensor,
        log_moneyness: torch.Tensor = None,
        moneyness: torch.Tensor = None,
        spot: torch.Tensor = None,
        strike: torch.Tensor = None,
        create_graph=False,
    ) -> torch.Tensor:
        """
        Returns Black-Scholes delta of the derivative.

        Shapes
        ------
        volatility : :math:`(*)`
        expiry : :math:`(*)`
        log_moneyness : :math:`(*)`
        moneyness : :math:`(*)`
        spot : :math:`(*)`
        strike : :math:`(*)`

        Examples
        --------
        >>> volatility = torch.tensor([0.18, 0.20, 0.22])
        >>> moneyness = torch.ones(3)
        >>> expiry = torch.ones(3)
        >>> BSEuropeanOption().gamma(
        ...     volatility=volatility, moneyness=moneyness, expiry=expiry, strike=1.0)
        tensor([2.2074, 1.9848, 1.8024])
        """
        spot = _parse_spot(
            spot=spot, strike=strike, moneyness=moneyness, log_moneyness=log_moneyness
        ).requires_grad_()
        delta = self.delta(
            volatility=volatility,
            spot=spot,
            strike=strike,
            expiry=expiry,
            create_graph=True,
        )
        return torch.autograd.grad(
            delta,
            inputs=spot,
            grad_outputs=torch.ones_like(spot),
            create_graph=create_graph,
        )[0]
