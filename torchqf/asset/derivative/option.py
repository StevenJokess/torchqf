from enum import Enum

from ...functional import european_payoff


class OptionStyle(Enum):
    EUROPEAN = "European"
    LOOKBACK = "Lookback"

    @classmethod
    def get_class(cls, style):
        """
        style : str or OptionStyle
        """
        return {cls.EUROPEAN: EuropeanOption, cls.LOOKBACK: LookbackOption}[cls(style)]


class Option:
    """
    Factory class

    Parameters
    ----------
    style : str
        Specifies option style.
    **kwargs
        Parameters used to initialize the specific option.

    Examples
    --------
    >>> Option(OptionStyle.EUROPEAN, ...)
    EuropeanOption(...)
    >>> Option("European", ...)
    EuropeanOption(...)
    """

    def __init__(self, style, *args, **kwargs):
        self.__class__ = OptionStyle.get_class(style)
        self.__init__(self, *args, **kwargs)


class EuropeanOption:
    """
    European option.

    Parameters
    ----------
    underlier : Asset
    strike : float, default=1.0
    call : bool, default=True
    maturity : float, default=1.0
    """

    def __init__(self, underlier, strike=1.0, call=True, maturity=1.0):
        self.underlier = underlier
        self.strike = strike
        self.call = call
        self.maturity = maturity

    def __repr__(self):
        return "EuropeanOption(...)"

    def payoff(self, keepdim=False):
        """
        Return payoff

        Parameters
        ----------
        keepdim : bool, default=False

        Returns
        -------
        payoff : Tensor, shape (*)
        """
        return european_payoff(self.underlier.price, keepdim=keepdim)


class LookbackOption:
    pass
