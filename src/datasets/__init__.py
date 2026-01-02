"""Dataset factory helpers."""

from pathlib import Path

from src.datasets.brain_mri import BrainMRIDataset
from src.datasets.chestxray import ChestXrayDataset
from src.datasets.fundus import FundusDataset
from src.datasets.isic import ISICDataset
from src.datasets.base import BaseDataset


def get_dataset(name: str, split: str = "test", data_dir: Path | str = Path("data")) -> BaseDataset:
	"""Return a dataset instance by name.

	Args:
		name: dataset key ('brain_mri', 'chestxray', 'fundus', 'isic')
		split: 'train'|'val'|'test'
		data_dir: base data directory (Path or str)
	"""
	name = name.lower()
	data_dir = Path(data_dir)
	if name == "brain_mri":
		return BrainMRIDataset(root_dir=str(data_dir / "raw" / "brain_mri"), split=split)
	if name == "chestxray":
		return ChestXrayDataset(root_dir=str(data_dir / "raw" / "chestxray"), split=split)
	if name == "fundus":
		return FundusDataset(root_dir=str(data_dir / "raw" / "fundus"), split=split)
	if name == "isic":
		# Note: stored under raw/isic for certification inputs
		return ISICDataset(root_dir=str(data_dir / "raw" / "isic"), split=split)
	raise ValueError(f"Unknown dataset: {name}")


__all__ = [
	"get_dataset",
	"BrainMRIDataset",
	"ChestXrayDataset",
	"FundusDataset",
	"ISICDataset",
]
