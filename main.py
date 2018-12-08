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
from util.misc import get_model, get_loss
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
		transforms=args.train_transforms, channels=args.img_channels, debug=False)

	val_dataset  = ProteinImageDataset(split="val", args=args,
		transforms=args.test_transforms, channels=args.img_channels, debug=False, n_samples=args.n_val_samples)

	test_dataset = ProteinImageDataset(split="test", args=args,
		transforms=args.test_transforms, channels=args.img_channels, debug=False)

	# dataloaders

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
		batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

	val_loader   = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=1, 
		num_workers=args.workers, pin_memory=True)
	
	test_loader  = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
		num_workers=args.workers, pin_memory=True)

	# model
	model = get_model(args).cuda()
	model = nn.DataParallel(model, device_ids=args.device_ids)
	model.to(primary_device)

	# training
	loss_func = get_loss(args, train_dataset.class_weights).to(primary_device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

	logger = Logger()
	max_score = 0

	for epoch in range(1, args.epochs+1):
		logger.print("Epoch {}".format(epoch))
		train(model, train_loader, loss_func, optimizer, logger)
		score = evaluate(model, val_loader, loss_func, logger)
		logger.save()
		if score > max_score:
			logger.save_model(model.module, epoch)
			max_score = score

	logger.print()
	logger.print("Test")
	test_results = test(model, test_loader)
	logger.write_test_results(test_results, test_dataset.test_ids)
	logger.save()
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

def evaluate(model, val_loader, loss_func, logger):
	model.eval()

	losses = []
	preds = []
	targets = []

	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):

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

	logger.log_eval({"loss": loss, "acc": acc, "f1": f1})
	return f1

def test(model, test_loader):
	model.eval()

	preds = []
	with torch.no_grad():
		for i, (image, frame_id) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):

			image = image.to(primary_device, dtype=torch.float32, non_blocking=True)
			output = model(image)
			output = torch.sigmoid(output)
			output = output.cpu().numpy()
			
			pred = postprocess(output)
			pred = test_loader.dataset.from_onehot(pred)

			frame_id = frame_id[0]
			preds.append((frame_id, pred))

	return preds

if __name__ == "__main__":
	main()
