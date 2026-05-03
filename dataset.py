import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
class IemocapMelDataset(Dataset):
	def __init__(self, h5_path: Path, rows: list[dict[str, str]]) -> None:
		self.h5_path = h5_path
		self.rows = rows
		self._h5 = None

	def _get_h5(self) -> h5py.File:
		if self._h5 is None:
			self._h5 = h5py.File(self.h5_path, "r")
		return self._h5

	def __len__(self) -> int:
		return len(self.rows)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		row = self.rows[idx]
		h5f = self._get_h5()
		mel = h5f[row["key"]][()]
		x = torch.tensor(mel, dtype=torch.float32)
		y = torch.tensor(int(row["label_id"]), dtype=torch.long)
		return x, y


def collate_pad(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
	xs, ys = zip(*batch)
	n_mels = xs[0].shape[0]
	max_t = max(x.shape[1] for x in xs)

	x_pad = torch.zeros(len(xs), n_mels, max_t, dtype=torch.float32)
	for i, x in enumerate(xs):
		t = x.shape[1]
		x_pad[i, :, :t] = x
	y = torch.stack(ys)
	return x_pad, y