import os
import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F

from loaders.loader import ProteinImageDataset
from models.resnet import Resnet
from models.pretrained import Pretrained
from models.loss import MultiLabelFocalLoss
from util.logger import Logger
from util.misc import get_model, get_loss, get_train_sampler, get_scheduler
from models.postprocess import postprocess

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from args import Args
args = Args()

primary_device = torch.device("cuda:{}".format(args.device_ids[0]))

def main():

	# datasets

	train_dataset = ProteinImageDataset(split="train", args=args,
		transforms=args.train_augmentation, channels=args.img_channels, debug=False)

	train_static_dataset = ProteinImageDataset(split="train", args=args,
		transforms=None, channels=args.img_channels, debug=False)

	val_dataset  = ProteinImageDataset(split="val", args=args,
		transforms=None, channels=args.img_channels, debug=False, n_samples=args.n_val_samples)

	# sampling
	train_sampler = get_train_sampler(args, train_dataset)
	shuffle = (train_sampler is None)

	# dataloaders

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, sampler=train_sampler,
		batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

	train_static_loader = torch.utils.data.DataLoader(train_static_dataset, shuffle=False,
		batch_size=1, num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=1, 
		num_workers=args.workers, pin_memory=True)

	# model
	model = get_model(args).cuda()
	model = nn.DataParallel(model, device_ids=args.device_ids)
	model.to(primary_device)

	# training
	loss_func = get_loss(args, train_dataset.class_weights).to(primary_device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
	scheduler = get_scheduler(args, optimizer)

	logger = Logger()
	max_score = 0

	for epoch in range(1, args.epochs+1):
		logger.print("Epoch {}".format(epoch))
		scheduler.step()
		train(model, train_loader, loss_func, optimizer, logger)
		score = evaluate(model, val_loader, loss_func, logger, splitname="val")
		evaluate(model, train_static_loader, loss_func, logger, splitname="train")
		logger.save()
		if score > max_score:
			logger.save_model(model.module, epoch)
			max_score = score

	logger.save()
	logger.print()
	logger.print("Test")
	logger.save_model(model, "final")
	logger.run_test("final")


def train(model, train_loader, loss_func, optimizer, logger):
	model.train()

	losses = []
	for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

		images = images.to(primary_device, dtype=torch.float32, non_blocking=True)
		labels = labels.to(primary_device, dtype=torch.float32, non_blocking=True)

		outputs = model(images)

		loss = loss_func(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		logger.log_loss(loss.item())
		if i % (len(train_loader)//args.log_freq) == 0:
			mean_loss = np.mean(logger.losses[-10:])
			tqdm.tqdm.write("Train loss: {}".format(mean_loss))
			logger.log("Train loss: {}".format(mean_loss))

def evaluate(model, loader, loss_func, logger, splitname="val"):
	model.eval()

	losses = []
	preds = []
	targets = []

	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True)
			labels = labels.to(primary_device, dtype=torch.float32, non_blocking=True)

			outputs = model(images)
			loss = loss_func(outputs, labels).item()

			pred = torch.sigmoid(outputs)
			pred = pred.cpu().numpy()
			pred = postprocess(pred)

			labels = labels.cpu().numpy().astype(np.int)

			losses.append(loss)
			preds.append(pred)
			targets.append(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds, average="macro")
	f1_perclass = metrics.f1_score(targets, preds, average=None)
	loss = np.mean(losses)

	logger.print()
	logger.print("Eval")
	logger.print("Loss:", loss)
	logger.print("Accuracy:", acc)
	logger.print("Macro F1:", f1)
	logger.print("Per-Class F1:", f1_perclass)
	logger.print()

	logger.log_eval({f"{splitname}-loss": loss, f"{splitname}-acc": acc, f"{splitname}-f1": f1})
	return f1

if __name__ == "__main__":
	main()
