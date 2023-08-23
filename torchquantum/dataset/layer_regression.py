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

from torchpack.datasets.dataset import Dataset
import numpy as np
import torch

__all__ = ["LayerRegression"]


class LayerRegressionDatasetV:
    """Layer Regression Dataset with Variational Input"""

    def __init__(self):
        """Initializes a LayerRegressionDatasetV object.

        Example:
            >>> dataset = LayerRegressionDatasetV()
        """

        self.data = None
        self.n_instance = 10000
        self.input = np.random.rand(5).astype(np.float) * 2 * np.pi
        self.output = np.random.rand(5).astype(np.float) * 2 - 1

    def __getitem__(self, index: int):
        """Gets an instance from the dataset.

        Args:
            index (int): The index of the instance.

        Returns:
            dict: A dictionary containing the input and output.

        Example:
            >>> dataset = LayerRegressionDatasetV()
            >>> instance = dataset[0]
        """
        
        instance = {"input": self.input, "output": self.output}
        return instance

    def __len__(self) -> int:
        """Returns the number of instances in the dataset.

        Returns:
            int: The number of instances.

        Example:
            >>> dataset = LayerRegressionDatasetV()
            >>> length = len(dataset)
        """
        
        return self.n_instance


class LayerRegressionV(Dataset):
    """Layer Regression Dataset with Variational Input"""
    
    def __init__(
        self,
    ):
        """Initializes a LayerRegressionV object.

        Example:
            >>> dataset = LayerRegressionV()
        """
        
        super().__init__(
            {split: LayerRegressionDatasetV() for split in ["train", "valid", "test"]}
        )


class LayerRegressionDataset:
    """Layer Regression Dataset
    
    Attributes:
        data: The dataset.
        n_instance (int): The number of instances.
        output: The output data.
    """
    
    def __init__(self):
        """Initializes a LayerRegressionDataset object.

        Example:
            >>> dataset = LayerRegressionDataset()
        """
        
        self.data = None
        self.n_instance = 10000
        mat = torch.randn((2**5, 2**5), dtype=torch.complex64)
        mat = mat.svd()[0].data

        self.output = mat

    def __getitem__(self, index: int):
        """Gets an instance from the dataset.

        Args:
            index (int): The index of the instance.

        Returns:
            dict: A dictionary containing the input and output.

        Example:
            >>> dataset = LayerRegressionDataset()
            >>> instance = dataset[0]
        """
        
        instance = {"input": self.output, "output": self.output}
        return instance

    def __len__(self) -> int:
        """Returns the number of instances in the dataset.

        Returns:
            int: The number of instances.

        Example:
            >>> dataset = LayerRegressionDataset()
            >>> length = len(dataset)
        """
        
        return self.n_instance


class LayerRegression(Dataset):
    """Layer Regression Dataset"""
    
    def __init__(
        self,
    ):
        """Initializes a LayerRegression object.

        Example:
            >>> dataset = LayerRegression()
        """
        
        super().__init__(
            {split: LayerRegressionDataset() for split in ["train", "valid", "test"]}
        )
