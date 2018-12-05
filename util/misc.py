import torch
from torch import nn

from models.resnet import Resnet
from models.pretrained import Pretrained
from models.loss import MultiLabelFocalLoss

def get_model(args):
	if args.arch in ["resnet152"]:
		model = Resnet()
	elif args.arch in ["inceptionv4", "setnet154"]:
		model = Pretrained(args.arch)
	else:
		raise ValueError("Invalid model architecture: {}".format(args.arch))
	return model

def get_loss(args, weights):

	if "inverse" in args.weight_mode:
		class_weights = weights
		if "sqrt" in args.weight_mode:
			class_weights = torch.sqrt(class_weights)
	else:
		class_weights = None

	if args.loss == "softmargin":
		loss_func = nn.MultiLabelSoftMarginLoss(weight=class_weights)
	elif args.loss == "focal":
		loss_func = MultiLabelFocalLoss(weight=class_weights)
	else:
		raise ValueError("Invalid loss function specifier: {}".format(args.loss))

	return loss_func