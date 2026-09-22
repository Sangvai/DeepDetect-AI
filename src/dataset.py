"""PyTorch Dataset for the preprocessed (already face-cropped) real/fake folders."""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import get_train_transforms, get_eval_transforms

# Fixed label convention used throughout the whole project: 0 = REAL, 1 = MANIPULATED.
# Keeping this explicit (instead of relying on alphabetical folder order, which
# ImageFolder would flip) is what makes "sigmoid output = P(manipulated)" correct.
CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_NAME = {0: "REAL", 1: "MANIPULATED"}


class FaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, float(label)


def get_dataloaders(processed_dir: str, batch_size: int = 32, num_workers: int = 0):
    train_ds = FaceDataset(os.path.join(processed_dir, "train"), get_train_transforms())
    val_ds = FaceDataset(os.path.join(processed_dir, "val"), get_eval_transforms())
    test_ds = FaceDataset(os.path.join(processed_dir, "test"), get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
