import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	
	def __init__(self, in_channels, out_channels, kernel, stride):
		super(ConvBlock, self).__init__()

		self.layer = nn.Sequential(
					nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel, stride = stride),
					nn.BatchNorm1d(out_channels),
					nn.ReLU()
					)

	def forward(self, x):

		out = self.layer(x)

		return out


class CNN1D(nn.Module):
	
	def __init__(self, num_classes):
		super(CNN1D, self).__init__()

		self.layer = nn.Sequential(
					ConvBlock(16, 16, 10, 4),
					ConvBlock(16, 16, 5, 2),
					ConvBlock(16, 16, 5, 2),
					ConvBlock(16, 32, 5, 2),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(32, 32, 4, 1),
					ConvBlock(32, 32, 4, 1),
					ConvBlock(32, 64, 4, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(64, 64, 3, 1),
					ConvBlock(64, 64, 3, 1),
					ConvBlock(64, 128, 3, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(128, 128, 2, 1),
					ConvBlock(128, 128, 2, 1),
					ConvBlock(128, 256, 2, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2)
					)

		self.linear = nn.Sequential(
					nn.Linear(1280, 512),
					nn.ReLU(),
					nn.Dropout(0.5),
					nn.Linear(512, 128),
					nn.ReLU(),
					nn.Linear(128, num_classes)
					)

	def forward(self, x):

		batch_size = x.shape[0]
		out = self.layer(x)
		out = out.view(batch_size, -1)
		out = self.linear(out)

		return out


class CNN1D_F(nn.Module):
	
	def __init__(self, num_classes):
		super(CNN1D_F, self).__init__()

		self.layer = nn.Sequential(
					ConvBlock(16, 16, 10, 4),
					ConvBlock(16, 16, 5, 2),
					ConvBlock(16, 16, 5, 2),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(16, 16, 5, 2),
					ConvBlock(16, 16, 5, 2),
					ConvBlock(16, 32, 5, 2),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(32, 32, 4, 1),
					ConvBlock(32, 32, 4, 1),
					ConvBlock(32, 64, 4, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(64, 64, 3, 1),
					ConvBlock(64, 64, 3, 1),
					ConvBlock(64, 128, 3, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2),
					ConvBlock(128, 128, 2, 1),
					ConvBlock(128, 128, 2, 1),
					ConvBlock(128, 256, 2, 1),
					nn.MaxPool1d(kernel_size = 2, stride = 2)
					)

		self.linear = nn.Sequential(
					nn.Linear(1280, 512),
					nn.ReLU(),
					nn.Dropout(0.5),
					nn.Linear(512, 128),
					nn.ReLU(),
					nn.Linear(128, num_classes)
					)

	def forward(self, x):

		batch_size = x.shape[0]
		out = self.layer(x)
		out = out.view(batch_size, -1)
		out = self.linear(out)

		return out