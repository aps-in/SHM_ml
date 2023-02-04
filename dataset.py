import torch
import torch.utils.data as data

class Dataset(data.Dataset):

	def __init__(self, x, y):

		self.x = x
		self.y = y

	def __len__(self):

		return len(self.x)

	def __getitem__(self, idx):

		x_item = torch.tensor(self.x[idx]).double()
		y_item = torch.tensor(self.y[idx]).long()

		return x_item, y_item