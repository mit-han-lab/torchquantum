"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations
import pytest
from pytest import raises
import torch
from torchquantum import QuantumDevice
from torchquantum.encoding import StateEncoder


class TestStateEncoding:
    """Test class for State Encoder."""

    @pytest.mark.parametrize(
        "qdev",
        [{}, list(range(10)), None, 1, True],
    )
    def test_qdev(self, qdev):
        with raises(
                TypeError,
                match=r"The qdev input ([\s\S]*?) must be of the type tq\.QuantumDevice\.",
        ):
            encoder = StateEncoder()
            encoder(qdev, torch.rand(2, 2))

    @pytest.mark.parametrize(
        "wires, x",
        [(2, {}), (4, list(range(10))), (1, None), (10, True), (5, 1)]
    )
    def test_type_x(self, wires, x):
        with raises(
                TypeError,
                match=r"The x input ([\s\S]*?) must be of the type torch\.Tensor\.",
        ):
            qdev = QuantumDevice(wires)
            encoder = StateEncoder()
            encoder(qdev, x)

    @pytest.mark.parametrize(
        "wires, x",
        [(2, torch.rand(2, 7)), (4, torch.rand(1, 20)), (1, torch.rand(1, 10))],
    )
    def test_size(self, wires, x):
        with raises(
            ValueError,
            match=r"The size of tensors in x \(\d+\) must be less than or "
            r"equal to \d+ for a QuantumDevice with "
            r"\d+ wires\.",
        ):
            qdev = QuantumDevice(wires)
            encoder = StateEncoder()
            encoder(qdev, x)

    @pytest.mark.parametrize(
        "wires, x, x_norm",
        [
            (
                2,
                [[0.3211], [0.1947]],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ),
            (
                4,
                [
                    [
                        0.1287,
                        0.9234,
                        0.4864,
                        0.6410,
                        0.4804,
                        0.9749,
                        0.1846,
                        0.3128,
                        0.0897,
                        0.4703,
                    ]
                ],
                [
                    [
                        0.0736,
                        0.5279,
                        0.2781,
                        0.3665,
                        0.2747,
                        0.5574,
                        0.1056,
                        0.1788,
                        0.0513,
                        0.2689,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                        0.0000,
                    ]
                ],
            ),
            (1, [[0.7275, 0.3252]], [[0.9129, 0.4081]]),
        ],
    )
    def test_state_encoding(self, wires, x, x_norm):
        """
        Tests the state encoding performed
        by the StateEncoder class.
        """
        x, x_norm = torch.tensor(x), torch.tensor(x_norm)
        qdev = QuantumDevice(wires)
        encoder = StateEncoder()
        encoder(qdev, x)

        assert qdev.states.shape[0] == x.shape[0]
        assert qdev.states.reshape(x.shape[0], -1).shape == (x.shape[0], pow(2, wires))
        assert torch.allclose(qdev.states.reshape(x.shape[0], -1), x_norm.type(torch.complex64), atol=1e-3)
