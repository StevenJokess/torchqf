import torch


# TODO(simaki) better naming; parse/extract/...
def _parse_moneyness(
    spot: torch.Tensor = None,
    strike: torch.Tensor = None,
    moneyness: torch.Tensor = None,
    log_moneyness: torch.Tensor = None,
    *_,
    **__,
) -> torch.Tensor:
    if moneyness is not None:
        return moneyness
    elif log_moneyness is not None:
        return log_moneyness.exp()
    else:
        return spot / strike


def _parse_log_moneyness(
    spot: torch.Tensor = None,
    strike: torch.Tensor = None,
    moneyness: torch.Tensor = None,
    log_moneyness: torch.Tensor = None,
    *_,
    **__,
) -> torch.Tensor:
    if log_moneyness is not None:
        return log_moneyness
    else:
        return _parse_moneyness(spot=spot, strike=strike, moneyness=moneyness).log()


def _parse_spot(
    spot: torch.Tensor = None,
    strike: torch.Tensor = None,
    moneyness: torch.Tensor = None,
    log_moneyness: torch.Tensor = None,
    *_,
    **__,
) -> torch.Tensor:
    if spot is not None:
        return spot
    elif moneyness is not None:
        return moneyness * strike
    else:
        return log_moneyness.exp() * strike


def root_bisect(
    f,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    precision: float = 1e-10,  # TODO(simaki) determine sensible default precision
    max_iter: int = 10000,  # TODO(simaki) determine sensible default iter; -log2(precision)
    differentiable: bool = False,
) -> torch.Tensor:
    """Find root by binary search assuming f is monotone.

    .. :math:

        f(output) = target

    Args:
        f : callable[[Tensor], Tensor]
        target : Tensor
    """
    assert not differentiable

    lower, upper = map(torch.as_tensor, (lower, upper))

    if not (lower < upper).all():
        raise ValueError("condition lower < upper should be satisfied.")

    if (f(lower) > f(upper)).all():
        # If decreasing function
        mf = lambda input: -f(input)
        return root_bisect(
            mf, -target, lower, upper, precision=precision, max_iter=max_iter
        )

    input_l = torch.full_like(target, lower)
    input_u = torch.full_like(target, upper)

    n_iter = 0
    while torch.max(input_u - input_l) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(f"Aborting since iteration exceeds abort={max_iter}.")

        input_m = (input_l + input_u) / 2
        output = f(input_m)
        input_l = torch.where(output <= target, input_m, input_l)
        input_u = torch.where(output > target, input_m, input_u)

    return input_u
