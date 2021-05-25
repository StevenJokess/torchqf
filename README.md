<h1 align="center">TorchQF: PyTorch Quant Finance</h1>

[![python versions](https://img.shields.io/pypi/pyversions/torchqf.svg)](https://pypi.org/project/torchqf)
[![version](https://img.shields.io/pypi/v/torchqf.svg)](https://pypi.org/project/torchqf)
[![CI](https://github.com/simaki/torchqf/actions/workflows/ci.yml/badge.svg)](https://github.com/simaki/torchqf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/simaki/torchqf/branch/main/graph/badge.svg)](https://codecov.io/gh/simaki/torchqf)
[![dl](https://img.shields.io/pypi/dm/torchqf)](https://pypi.org/project/torchqf)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyTorch-based library for Quantitative Finance.

## Installation

```sh
pip install torchqf
```

## Features

### Autogreek

```py
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
```

## Examples

...

