import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from models.resnet import Resnet
from models.pretrained import Pretrained
from models.loss import MultiLabelFocalLoss, FBetaLoss

def get_model(args):
	n_channels = len(set(args.img_channels))
	if args.arch in ["resnet152"]:
		model = Resnet(n_input_channels=n_channels)
	elif args.arch in ["inceptionv4", "setnet154"]:
		model = Pretrained(args.arch, n_input_channels=n_channels)
	else:
		raise ValueError("Invalid model architecture: {}".format(args.arch))
	return model

def get_loss(args, weights):

	if args.weight_method == "loss":
		if args.weight_mode is not None and "inverse" in args.weight_mode:
			class_weights = weights
			if "sqrt" in args.weight_mode:
				class_weights = torch.sqrt(class_weights)
		else:
			class_weights = None
	else:
		class_weights = None

	if args.loss == "softmargin":
		loss_func = nn.MultiLabelSoftMarginLoss(weight=class_weights)
	elif args.loss == "focal":
		loss_func = MultiLabelFocalLoss(weight=class_weights, gamma=args.focal_gamma)
	elif args.loss == "fbeta":
		loss_func = FBetaLoss(weight=class_weights, beta=args.fbeta, soft=True)
	else:
		raise ValueError("Invalid loss function specifier: {}".format(args.loss))

	return loss_func

def get_train_sampler(args, dataset):
	if args.weight_method == "sampling":
		return WeightedRandomSampler(weights=dataset.example_weights, num_samples=len(dataset))
	else:
		return None