from torchpack.datasets.dataset import Dataset


__all__ = ["VQE"]


class VQEDataset:
    """Dataset for VQE.
    
    Attributes:
        split (str): Split of the dataset ("train", "valid", "test").
        steps_per_epoch (int): Number of steps per epoch.
    
    Methods:
        __getitem__: Get an item from the dataset
        __len__: Get the length of the dataset
        
    Examples:
        >>> dataset = VQEDataset(split='train', steps_per_epoch=100)
        >>> instance = dataset[0]
    """
    
    def __init__(self, split, steps_per_epoch):
        """Initialize the VQEDataset.

        Args:
            split (str): Split of the dataset ("train", "valid", "test")
            steps_per_epoch (int): Number of steps per epoch
        """
        
        self.split = split
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index: int):
        """Get an instance from the dataset.
        
        Args:
            index (int): Index of the item

        Returns:
            dict: instance containing the input and target
        
        Examples:
            >>> instance = dataset[0]
        """
        
        instance = {"input": -1, "target": -1}

        return instance

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: length of the dataset
            
        Examples:
            >>> length = len(dataset)
        """
        
        if self.split == "train":
            return self.steps_per_epoch
        else:
            return 1


class VQE(Dataset):
    """Dataset for the VQE.

    Attributes:
        steps_per_epoch (int): Number of steps per epoch

    Methods:
        __init__: Initialize the VQE dataset
    
    Examples:
        >>> dataset = VQE(steps_per_epoch=100)
    """
    
    def __init__(self, steps_per_epoch):
        """Initialize the VQE dataset.

        Args:
            steps_per_epoch (int): Number of steps per epoch
        """
        
        super().__init__(
            {
                split: VQEDataset(split=split, steps_per_epoch=steps_per_epoch)
                for split in ["train", "valid", "test"]
            }
        )
