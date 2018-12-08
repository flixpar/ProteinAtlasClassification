import torch
from torch import nn

class BinaryFocalLoss(nn.Module):

	def __init__(self, gamma=2.0, weight=None, reduction="elementwise_mean"):
		super(BinaryFocalLoss, self).__init__()
		if weight is not None: self.register_buffer('weight', weight)
		self.weight = weight
		self.reduction = reduction
		self.gamma = gamma
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
		self.gamma = gamma
		self.eps = 1e-7

	def forward(self, input, target):

		input = torch.sigmoid(input)
		input = input.clamp(self.eps, 1.0-self.eps)
		loss = - (target * torch.pow(1 - input, self.gamma) * torch.log(input)) - ((1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))

		if self.weight is not None:
			loss = torch.mul(loss, self.weight)

		loss = torch.mean(loss, dim=1)

		if self.reduction == "elementwise_mean":
			loss = loss.mean(dim=0)
		elif self.reduction == "sum":
			loss = loss.sum(dim=0)

		return loss

class FBetaLoss(nn.Module):

	def __init__(self, beta=1, soft=True, weight=None, reduction="elementwise_mean"):
		super(FBetaLoss, self).__init__()
		if weight is not None: self.register_buffer('weight', weight)
		self.weight = weight
		if reduction != "elementwise_mean": raise NotImplementedError()
		self.reduction = reduction
		self.beta = beta
		self.eps = 1e-7
		self.soft = soft

	def forward(self, input, target):

		input = torch.sigmoid(input)
		target = target.float()

		if not self.soft:
			input = (input >= 0.5).float()

		tp = torch.sum(input * target,             dim=0)
		tn = torch.sum((1 - input) * (1 - target), dim=0)
		fp = torch.sum(input * (1 - target),       dim=0)
		fn = torch.sum((1 - input) * target,       dim=0)

		precision = tp / (tp + fp + self.eps)
		recall    = tp / (tp + fn + self.eps)

		beta2 = self.beta**2
		loss = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall + self.eps)

		if self.weight is not None:
			loss = torch.mul(loss, self.weight)

		loss = 1 - loss.mean()
		return loss
