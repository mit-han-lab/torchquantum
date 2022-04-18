import torch
import torch.nn.functional as F


def complex_mse(output, target):
    return F.mse_loss(torch.view_as_real(output),
                      torch.view_as_real(target))


def complex_mae(output, target):
    return (torch.view_as_real(output)
            - torch.view_as_real(target)).abs().mean()


def minimize(output, target):
    return output.sum()


def maximize(output, target):
    return -output.sum()
