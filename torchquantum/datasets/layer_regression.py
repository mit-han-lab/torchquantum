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
