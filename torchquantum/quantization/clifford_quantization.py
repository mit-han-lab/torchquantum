import torch
import torchquantum as tq
import numpy as np


class CliffordQuantizer(object):
    def __init__(self):
        pass

    def quantize(self, model: torch.nn.Module):
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (tq.RX, tq.RY, tq.RZ)):
                    param = module.params[0][0].item()
                    # now the param is from 0 to 2*pi
                    param = param % (2 * np.pi)
                    # print(f'before {param}')
                    param = np.pi / 2 * np.floor(param / (np.pi / 2))
                    # print(f'after {param}')
                    module.params[0][0] = param
        return model


