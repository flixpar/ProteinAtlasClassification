import torch
from torch import nn
import torchvision
import pretrainedmodels

class Pretrained(nn.Module):

	def __init__(self, arch="resnet152", n_classes=28):
		super(Pretrained, self).__init__()

		valid_model_sizes = {
			"resnet152": 2048,
			"inceptionv4": 1536
		}
		valid_models = list(valid_model_sizes.keys())

		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})
		
		self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet').features
		self.feat_size = valid_model_sizes[arch]

		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Linear(self.feat_size, n_classes)

		nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.net(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

