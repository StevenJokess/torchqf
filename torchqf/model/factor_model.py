import torch
from torch.nn import Module


class FactorModel(Module):
    """
    Factor model.

    Attributes
    ----------
    beta : Tensor
        Created after `fit`.

    Shape
    -----
    input : :math:`(*, A, T)`
        :math:`A` is the number of assets.
        :math:`T` is the number of time steps.
    output : :math:`(*, A, T)`
        :math:`A` is the number of assets.
        :math:`T` is the number of time steps.
    factor : :math:`(*, F, T)`
        :math:`F` is the number of assets.
        :math:`T` is the number of time steps.
    beta : :math:`(F, A)`
        :math:`F` is the number of factor.
        :math:`A` is the number of assets.

    Examples
    --------
    >>> import torchqf
    >>> from torchqf.stochastic import generate_brownian

    >>> _ = torch.manual_seed(42)

    >>> input = generate_brownian((2, 5), 0.1)
    >>> factor = generate_brownian((2, 5), 0.1)
    >>> fm = FactorModel().fit(input, factor)
    >>> fm(input, factor)
    tensor([[ 0.0032, -0.0023, -0.0046,  0.0138, -0.0099],
            [-0.0234,  0.0157, -0.0017, -0.0052,  0.0025]])

    The method `fit_forward` fits and forwards at once.

    >>> fm.fit_forward(input, factor)
    tensor([[ 0.0032, -0.0023, -0.0046,  0.0138, -0.0099],
            [-0.0234,  0.0157, -0.0017, -0.0052,  0.0025]])
    """

    def fit(self, input, factor):
        """
        Fit the model using the asset returns (`input`) and the factor returns (`factor`).

        Parameters
        ----------
        input : Tensor
            Returns of assets.
        factor : Tensor
            Returns of factors.

        Returns
        -------
        self : Module

        Shape
        -----
        input : :math:`(*, A, T)`
            :math:`A` is the number of assets.
            :math:`T` is the number of time steps.
        factor : :math:`(*, F, T)`
            :math:`F` is the number of assets.
            :math:`T` is the number of time steps.
        beta : :math:`(*, F, A)`
            :math:`F` is the number of factors.
            :math:`A` is the number of assets.
        """
        assert input.size(-1) == factor.size(-1), "numbers of time steps do not match"
        assert input.ndim == 2, "not supported"
        assert factor.ndim == 2, "not supported"

        X = factor.t()  # shape : (T, F)
        y = input.transpose(-2, -1)  # shape : (T, A)

        # Compute beta : (F, T) @ (T, A) = (F, A)
        self.beta = torch.mm(torch.linalg.pinv(X), y)

        return self

    def forward(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        factor_return = torch.mm(self.beta.t(), factor)  # shape : (*, A, T)
        return input - factor_return

    def fit_forward(self, input: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        return self.fit(input, factor)(input, factor)
