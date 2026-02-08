import torch
from hs_tasnet import HSTasNet


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def report(name, model):
    params = count_params(model)
    fp32_mb = params * 4 / (1024 ** 2)
    fp16_mb = params * 2 / (1024 ** 2)
    print(f"{name} params: {params:,}")
    print(f"{name} weights fp32: {fp32_mb:.1f} MB")
    print(f"{name} weights fp16: {fp16_mb:.1f} MB")


def main():
    small = HSTasNet(small=True)
    large = HSTasNet(small=False)

    report("small", small)
    report("large", large)


if __name__ == "__main__":
    main()
