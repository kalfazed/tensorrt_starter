import torch
from cuda_ops.add_scalar import add_scalar


def main():
    tensor = torch.randn(10, device='cuda')
    scalar = 0.5

    print(f"Before:\n{tensor}")
    # Apply the custom CUDA operation
    add_scalar.add_scalar_d(tensor, scalar)
    print(f"After:\n{tensor}")


if __name__ == '__main__':
    main()
