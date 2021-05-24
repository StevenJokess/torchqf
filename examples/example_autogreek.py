import sys

import torch

if __name__ == "__main__":
    sys.path.append("..")

    import torchqf

    pricing_formula = torchqf.bs.BSEuropeanOption().price

    delta = torchqf.autogreek.delta(
        pricing_formula,
        volatility=torch.tensor([0.18, 0.20, 0.22]),
        spot=torch.ones(3),
        expiry=torch.ones(3),
    )
    print(delta)
    # tensor([0.5359, 0.5398, 0.5438])
