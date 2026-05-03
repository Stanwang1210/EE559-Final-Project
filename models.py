import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanTimeClassifier(nn.Module):
	def __init__(self, n_mels: int, hidden_dim: int, num_classes: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_mels, hidden_dim*2),
			nn.ReLU(),
			nn.Dropout(p=0.1),
            nn.Linear(hidden_dim*2, hidden_dim ),
			nn.ReLU(),
			nn.Dropout(p=0.1),
   			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Dropout(p=0.1),
			nn.Linear(hidden_dim // 2, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, n_mels, T] -> mean over time axis -> [B, n_mels]
		x = self.net(x.transpose(1, 2)).transpose(1, 2)  # [B, T, n_mels] -> [B, T, hidden_dim] -> [B, T, num_classes]
		x_pred = x.mean(dim=-1)
		return x_pred


class CNNClassifier(nn.Module):
	def __init__(self, n_mels: int, hidden_dim: int, num_classes: int, temporal_pool_kernel: int = 2) -> None:
		super().__init__()
		d_model = hidden_dim * 2
		self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=temporal_pool_kernel, stride=temporal_pool_kernel),
            nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=temporal_pool_kernel, stride=temporal_pool_kernel),
        )
		
		self.fc = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Dropout(p=0.1),
			nn.Linear(hidden_dim // 2, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, n_mels, T]
		x = self.cnn(x)  # [B, hidden_dim*2, T]
		x = x.mean(dim=-1)  # [B, hidden_dim*2] - mean over time
		x_pred = self.fc(x)  # [B, num_classes]
		return x_pred



