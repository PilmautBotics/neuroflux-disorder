from PIL import Image
from torch.utils.data import Dataset

class NeurofluxDataset(Dataset):
    """Dataset for neuroflux medical images and classification labels."""
    def __init__(self, image_paths: list, labels: list, transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: Corresponding class labels
            transform: Image transformation process
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
