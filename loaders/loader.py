import os
import csv
import glob
import random
import numpy as np

import torch
import torchvision

import cv2
from PIL import Image
import albumentations as tfms

class ProteinImageDataset(torch.utils.data.Dataset):

	def __init__(self, split, args, transforms=None, channels="g", debug=False, n_samples=None):
		self.split = split
		self.transforms = transforms
		self.image_channels = channels
		self.debug = debug
		self.n_classes = 28
		self.resize = tfms.Resize(args.img_size, args.img_size) if args.img_size is not None else None
		self.base_path = args.datapath
		self.split_folder = os.path.join(self.base_path, "test" if self.split=="test" else "train")

		# check for valid image mode
		if not (set(self.image_channels) <= set("rgby")):
			raise ValueError("Invalid image channels selection.")

		# split the training set into training and validation
		if split in ["train", "val"]:
			with open(os.path.join(self.base_path, 'train.csv'), 'r') as f:
				csvreader = csv.reader(f)
				data = list(csvreader)[1:]
			ids = sorted([d[0] for d in data])
			train_ids = ids[:round(len(ids)*args.trainval_ratio)]
			val_ids = ids[round(len(ids)*args.trainval_ratio):]
			label_lookup = {k:np.array(v.split(' ')) for k,v in data}

		# construct dataset

		if self.split == "train":
			self.data = [(i, label_lookup[i]) for i in train_ids]

			labels = [self.to_onehot(np.array(v.split(' '))) for _, v in data]
			self.class_weights = np.sum(labels, axis=0)
			self.class_weights = self.class_weights.max() / self.class_weights
			self.class_weights = self.class_weights / self.n_classes
			self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

		elif self.split == "val":
			self.data = [(i, label_lookup[i]) for i in val_ids]

		elif self.split == "test":
			with open(os.path.join(self.base_path, 'sample_submission.csv'), 'r') as f:
				lines = list(csv.reader(f))[1:]
				test_ids = [line[0] for line in lines]
			self.data = [(i, None) for i in test_ids]
			self.test_ids = test_ids

		else:
			raise Exception("Invalid dataset split.")

		if n_samples is not None and n_samples < len(self.data):
			self.data = random.sample(self.data, n_samples)

		# debug
		if self.debug:
			self.data = self.data[:100]

	def __getitem__(self, index):

		example_id, label = self.data[index]

		if self.image_channels == "g":
			fn = os.path.join(self.split_folder, example_id + "_green.png")
			img = cv2.imread(fn, cv2.IMREAD_COLOR)

		elif set(self.image_channels) == set("rgb"):
			r = cv2.imread(os.path.join(self.split_folder, example_id + "_red.png"),   cv2.IMREAD_GRAYSCALE)
			g = cv2.imread(os.path.join(self.split_folder, example_id + "_green.png"), cv2.IMREAD_GRAYSCALE)
			b = cv2.imread(os.path.join(self.split_folder, example_id + "_blue.png"),  cv2.IMREAD_GRAYSCALE)
			img = np.stack([r, g, b], axis=-1)

		elif set(self.image_channels) == set("rgby"):
			r = cv2.imread(os.path.join(self.split_folder, example_id + "_red.png"),    cv2.IMREAD_GRAYSCALE)
			g = cv2.imread(os.path.join(self.split_folder, example_id + "_green.png"),  cv2.IMREAD_GRAYSCALE)
			b = cv2.imread(os.path.join(self.split_folder, example_id + "_blue.png"),   cv2.IMREAD_GRAYSCALE)
			y = cv2.imread(os.path.join(self.split_folder, example_id + "_yellow.png"), cv2.IMREAD_GRAYSCALE)
			img = np.stack([r, g, b, y], axis=-1)

		else:
			raise NotImplementedError("Image channel mode not yet supported.")

		if self.resize is not None:
			img = self.resize(image=img)["image"]
		if self.transforms is not None:
			img = self.transforms(image=img)["image"]
		img = torch.from_numpy(img.transpose((2, 0, 1)))

		if self.split in ["train", "val"]:
			label = self.to_onehot(label)
			return img, label
		else:
			return img, example_id

	def __len__(self):
		return len(self.data)

	def to_onehot(self, lbl):
		out = np.zeros(self.n_classes)
		for i in lbl:
			out[int(i)] = 1
		return out

	def from_onehot(self, lbl):
		return np.where(lbl.flatten() == 1)[0].tolist()

