
import torch
from torch.utils.data import Dataset


class SyntheticMethaneDataset(Dataset):
    """Generate fake images and matching binary plume masks."""

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 32,
        num_channels: int = 3,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels

        # A private generator makes dataset creation repeatable.
        generator = torch.Generator().manual_seed(seed)

        # Fake sensor images: [N, C, H, W]
        self.images = torch.rand(
            num_samples,
            num_channels,
            image_size,
            image_size,
            generator=generator,
        ) * 0.2

        # Binary masks: [N, 1, H, W]
        self.masks = torch.zeros(
            num_samples, 1, image_size, image_size
        )

        # Add a random square "plume" to each image and its mask.
        for i in range(num_samples):
            y = torch.randint(
                4, image_size - 8, (1,), generator=generator
            ).item()
            x = torch.randint(
                4, image_size - 8, (1,), generator=generator
            ).item()

            self.masks[i, 0, y:y + 5, x:x + 5] = 1.0

            # Make the fake plume brighter than its background.
            self.images[i, :, y:y + 5, x:x + 5] += 0.7

        self.images.clamp_(0.0, 1.0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.images[index], self.masks[index]
