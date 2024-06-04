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
from typing import Callable
import pytest
from pytest import raises
from unittest import mock
from qiskit.circuit import QuantumCircuit
import torch
from torchquantum import (
    QuantumDevice,
    GeneralEncoder,
    StateEncoder,
    PhaseEncoder,
    MultiPhaseEncoder,
)
from torchquantum.functional import func_name_dict


class TestGeneralEncoder:
    """Test class for General Encoder."""

    @pytest.mark.parametrize("func_list", [None, 1, 2.4, True, list(range(2))])
    def test_invalid_func_list(self, func_list):
        with raises(
            TypeError, match=r"The input func_list must be of the type list\[dict\]\."
        ):
            _ = GeneralEncoder(func_list)

    @pytest.mark.parametrize(
        "func_list", [[{"key1": 1}], [{"func": "rx"}], [{"func": "rx", "key2": None}]]
    )
    def test_func_list_keys(self, func_list):
        with raises(
            ValueError,
            match="The dictionary in func_list must contain the "
            "keys: input_idx, func, and wires.",
        ):
            _ = GeneralEncoder(func_list)

    @pytest.mark.parametrize(
        "wires, func_list",
        [
            (1, [{"input_idx": [0], "func": "ry", "wires": [0]}]),
            (
                2,
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                ],
            ),
            (
                4,
                [
                    {"input_idx": [0], "func": "rz", "wires": [0]},
                    {"input_idx": None, "func": "sx", "wires": [0]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ],
            ),
        ],
    )
    def test_general_encoding(self, wires, func_list):
        """Tests the GeneralEncoder class."""
        encoder = GeneralEncoder(func_list)
        qdev = QuantumDevice(wires)
        mock_func = mock.Mock()
        for func_dict in func_list:
            func = func_dict["func"]
            with mock.patch.dict(func_name_dict, {func: mock_func}):
                encoder(qdev, torch.rand(1, pow(2, wires)))
                assert mock_func.call_count >= 1

    @pytest.mark.parametrize(
        "batch_size, wires, func_list",
        [
            (2, 1, [{"input_idx": [0], "func": "rz", "wires": [0]}]),
            (
                4,
                2,
                [
                    {"input_idx": [0], "func": "ryy", "wires": [0, 1]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ],
            ),
            (
                2,
                4,
                [
                    {"input_idx": [0], "func": "rzz", "wires": [0, 2]},
                    {"input_idx": [1], "func": "rxx", "wires": [1, 2]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "rzx", "wires": [1, 3]},
                ],
            ),
        ],
    )
    def test_to_qiskit(self, batch_size, wires, func_list):
        """Tests conversion of GeneralEncoder to Qiskit."""
        x = torch.rand(batch_size, pow(2, wires))
        encoder = GeneralEncoder(func_list)
        qdev = QuantumDevice(n_wires=wires, bsz=batch_size)
        encoder(qdev, x)
        resulting_circuit = encoder.to_qiskit(wires, x)
        for circuit in resulting_circuit:
            assert isinstance(circuit, QuantumCircuit)

    @pytest.mark.parametrize(
        "batch_size, wires, func_list",
        [
            (2, 1, [{"input_idx": [0], "func": "hadamard", "wires": [0]}]),
            (2, 2, [{"input_idx": [0], "func": "xx", "wires": [0, 1]}]),
        ],
    )
    def test_not_implemeted_qiskit(self, batch_size, wires, func_list):
        """Tests conversion of GeneralEncoder to Qiskit."""
        x = torch.rand(batch_size, pow(2, wires))
        encoder = GeneralEncoder(func_list)
        qdev = QuantumDevice(n_wires=wires, bsz=batch_size)
        encoder(qdev, x)
        with raises(NotImplementedError, match=r"([\s\S]*?) is not supported yet\."):
            _ = encoder.to_qiskit(wires, x)


class TestPhaseEncoder:
    """Test class for Phase Encoder."""

    @pytest.mark.parametrize("func", [None, 1, 2.4, {}, True, list(range(2))])
    def test_func_type(self, func):
        """Test the type of func input"""
        with raises(TypeError, match="The input func must be of the type str."):
            _ = PhaseEncoder(func)

    @pytest.mark.parametrize("func", ["hadamard", "ry", "xx", "paulix", "i"])
    def test_phase_encoding(self, func):
        """Tests the PhaseEncoder class."""
        assert func in func_name_dict
        encoder = PhaseEncoder(func)
        qdev = QuantumDevice(2)
        with mock.patch.object(encoder, "func") as mock_func:
            encoder(qdev, torch.rand(2, 4))
            assert mock_func.call_count >= 1


class TestMultiPhaseEncoder:
    """Test class for Multi-phase Encoder."""

    @pytest.mark.parametrize(
        "wires, funcs",
        [
            (10, ["rx", "hadamard"]),
            (2, ["swap", "ry"]),
            (3, ["xx"]),
            (1, ["paulix", "i"]),
        ],
    )
    def test_invalid_func(self, wires, funcs):
        with raises(ValueError, match=r"The func (.*?) is not supported\."):
            encoder = MultiPhaseEncoder(funcs)
            qdev = QuantumDevice(n_wires=wires)
            encoder(qdev, torch.rand(1, pow(2, wires)))

    # NOTE: Test with func = u1 currently fails.
    @pytest.mark.parametrize(
        "batch_size, wires, funcs",
        [(2, 5, ["ry", "phaseshift"]), (1, 4, ["u2"]), (3, 1, ["u3"])],
    )
    def test_phase_encoding(self, batch_size, wires, funcs):
        """Tests the MultiPhaseEncoder class."""
        # wires = 4
        encoder = MultiPhaseEncoder(funcs)
        qdev = QuantumDevice(n_wires=wires, bsz=batch_size)
        mock_func = mock.Mock()
        for func in encoder.funcs:
            with mock.patch.dict(func_name_dict, {func: mock_func}):
                encoder(qdev, torch.rand(batch_size, pow(2, wires)))
                assert mock_func.call_count >= 1


class TestStateEncoder:
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
            encoder.forward(qdev, torch.rand(2, 2))

    @pytest.mark.parametrize(
        "wires, x", [(2, {}), (4, list(range(10))), (1, None), (10, True), (5, 1)]
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
        assert torch.allclose(
            qdev.states.reshape(x.shape[0], -1), x_norm.type(torch.complex64), atol=1e-3
        )
