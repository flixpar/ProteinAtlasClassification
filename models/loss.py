import torch
from torch import nn

class FocalLoss(nn.Module):

	def __init__(self, weight=None, reduction='elementwise_mean', n_classes=28):
		super(FocalLoss, self).__init__()
		raise NotImplementedError("Multi-class focal loss not yet implemented")

	def forward(self, x):
		raise NotImplementedError("Multi-class focal loss not yet implemented")

class BinaryFocalLoss(nn.Module):

	def __init__(self, gamma=2.0, weight=None, reduction="elementwise_mean"):
		super(BinaryFocalLoss, self).__init__()
		if weight is not None: self.register_buffer('weight', weight)
		self.weight = weight
		self.reduction = reduction
		self.eps = 1e-7

	def forward(self, input, target):
		
		input = input.clamp(self.eps, 1.0-self.eps)
		loss = - (target * torch.pow(1 - input, self.gamma) * torch.log(input)) - ((1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))

		if self.weight is not None:
			loss = torch.mul(loss, self.weight)

		if self.reduction == "elementwise_mean":
			loss = loss.mean(dim=0)
		elif self.reduction == "sum":
			loss = loss.sum(dim=0)

		return loss

class MultiLabelFocalLoss(nn.Module):

	def __init__(self, gamma=2.0, weight=None, reduction="elementwise_mean"):
		super(MultiLabelFocalLoss, self).__init__()
		if weight is not None: self.register_buffer('weight', weight)
		self.weight = weight
		self.reduction = reduction
		self.eps = 1e-7

	def forward(self, input, target):

		input = input.clamp(self.eps, 1.0-self.eps)
		loss = - (target * torch.pow(1 - input, self.gamma) * torch.log(input)) - ((1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))

		if self.weight is not None:
			loss = torch.mul(loss, self.weight)

		loss = torch.sum(loss, dim=1)

		if self.reduction == "elementwise_mean":
			loss = loss.mean(dim=0)
		elif self.reduction == "sum":
			loss = loss.sum(dim=0)

		return loss

