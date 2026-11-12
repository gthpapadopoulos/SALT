"""Exact paper sample units, mask handling, and flip augmentation."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


SPLITS = Path(__file__).resolve().parents[1] / "splits"


def read_split(dataset, split):
    with (SPLITS / dataset / f"{split}.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    paths = [row["image_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate samples in {dataset}/{split}")
    for row in rows:
        for key in ("image_path", "mask_path"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Split paths must be relative: {path}")
    return rows


def image_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class StructuralImages(Dataset):
    def __init__(self, data_root, dataset, split, resolution=518, augmented=False):
        self.root = Path(data_root) / dataset
        self.rows = read_split(dataset, split)
        self.resolution = resolution
        self.orientations = 4 if augmented else 1
        self.transform = image_transform(resolution)

    def __len__(self):
        return len(self.rows) * self.orientations

    def __getitem__(self, index):
        row = self.rows[index // self.orientations]
        orientation = index % self.orientations
        with Image.open(self.root / row["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        with Image.open(self.root / row["mask_path"]) as annotation:
            annotation = annotation.convert("L").resize(
                (self.resolution, self.resolution), Image.Resampling.NEAREST,
            )
            mask = (transforms.functional.pil_to_tensor(annotation) > 0).to(torch.uint8)
        if orientation & 1:
            image, mask = image.flip(-1), mask.flip(-1)
        if orientation & 2:
            image, mask = image.flip(-2), mask.flip(-2)
        return {"image": image, "mask": mask, "label": int(row["label"]), "filename": row["image_path"]}


def epoch_indices(image_count, epoch, seed):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(image_count, generator=generator)
    orientation = torch.randint(4, (image_count,), generator=generator)
    return (order * 4 + orientation).tolist()


def make_loader(dataset, config, indices=None):
    dataset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                      num_workers=config["workers"], pin_memory=True)
