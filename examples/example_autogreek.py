import sys

if __name__ == "__main__":
    sys.path.append("..")

    import torch

    import torchqf

    pricing_formula = torchqf.bs.BSEuropeanOption().price

    delta = torchqf.autogreek.delta(
        pricing_formula,
        volatility=torch.tensor([0.18, 0.20, 0.22]),
        spot=torch.ones(3),
        strike=torch.ones(3),
        expiry=torch.ones(3),
    )
    print(delta)
    # tensor([0.5359, 0.5398, 0.5438])

    gamma = torchqf.autogreek.gamma(
        pricing_formula,
        volatility=torch.tensor([0.18, 0.20, 0.22]),
        spot=torch.ones(3),
        strike=torch.ones(3),
        expiry=torch.ones(3),
    )
    print(gamma)
    # tensor([2.2074, 1.9848, 1.8024])
