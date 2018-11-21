import torch
from torch import nn
import torchvision

class Resnet(nn.Module):

	def __init__(self, n_classes=28):
		super(Resnet, self).__init__()
		base_model = torchvision.models.resnet152(pretrained=True)

		self.resnet = nn.Sequential(
			base_model.conv1,
			base_model.bn1,
			base_model.relu,
			base_model.maxpool,
			base_model.layer1,
			base_model.layer2,
			base_model.layer3,
			base_model.layer4
		)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Linear(2048, n_classes)

		nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.resnet(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

