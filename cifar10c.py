import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

MEAN = [0.491, 0.482, 0.447]
STD = [0.247, 0.243, 0.262]


class CorruptedCIFAR(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i]), i


def load_severity(root, severity, corruptions=None):
    names = corruptions or CORRUPTIONS
    labels = np.load(os.path.join(root, 'labels.npy'))[:10000]
    xs, ys = [], []
    lo = (severity - 1) * 10000
    for name in names:
        path = os.path.join(root, name + '.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        xs.append(np.load(path)[lo:lo + 10000])
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def get_loader(root, severity, batch_size=128, corruptions=None, num_workers=4):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    x, y = load_severity(root, severity, corruptions)
    return DataLoader(CorruptedCIFAR(x, y, tf), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)
