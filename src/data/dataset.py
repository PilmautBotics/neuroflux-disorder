from PIL import Image
from torch.utils.data import Dataset


class NeurofluxDataset(Dataset):
    """Dataset for neuroflux medical images and classification labels."""

    def __init__(self, image_paths: list, labels: list, transform=None):
        """Initialize the dataset.

        Args:
            image_paths (list): List of paths to image files.
            labels (list): Corresponding class labels.
            transform (callable, optional): Image transformation process. Defaults to None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image and
                  label is the corresponding class label.
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
