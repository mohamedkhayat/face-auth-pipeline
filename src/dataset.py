from torch.utils.data import Dataset
import random
from PIL import Image


class FaceVerificationDataset(Dataset):
    def __init__(self, label_to_imgs, transform=None):
        self.label_to_imgs = label_to_imgs
        self.transform = transform
        self.labels = list(label_to_imgs.keys())

        self.valid_pairs = [
            lbl for lbl in self.labels if len(self.label_to_imgs[lbl]) >= 2
        ]

    def __len__(self):
        return len(self.valid_pairs) * 10

    def __getitem__(self, idx):
        anchor_label = random.choice(self.valid_pairs)

        anchor_path, positive_path = random.sample(self.label_to_imgs[anchor_label], 2)

        negative_candidates = [lbl for lbl in self.valid_pairs if lbl != anchor_label]
        negative_label = random.choice(negative_candidates)
        negative_path = random.choice(self.label_to_imgs[negative_label])

        try:
            anchor_img = self.load_and_transform(anchor_path)
            positive_img = self.load_and_transform(positive_path)
            negative_img = self.load_and_transform(negative_path)

            if (anchor_img is None) or (positive_img is None) or (negative_img is None):
                return self.__getitem__(random.randint(0, len(self) - 1))

            return anchor_img, positive_img, negative_img

        except Exception as e:
            print(f"Error loading images : {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def load_and_transform(self, img_path):
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class TSNEDataset(Dataset):
    def __init__(self, label_to_imgs, transform=None):
        self.transform = transform
        self.samples = []
        for label, image_paths in label_to_imgs.items():
            for path in image_paths:
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
