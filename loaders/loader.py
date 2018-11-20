import os
import csv
import glob
import numpy as np

import torch
import torchvision
from torchvision import transforms as tfms

import cv2
from PIL import Image

class ProteinImageDataset(torch.utils.data.Dataset):

	def __init__(self, split="train", transforms=tfms.ToTensor(), debug=False):
		self.split = split
		self.transforms = transforms
		self.n_classes = 28

		self.base_path = "/home/felix/projects/class/deeplearning/final/data/"

		# split the training set into training and validation
		if split in ["train", "val"]:
			with open(os.path.join(self.base_path, 'train.csv'), 'r') as f:
				csvreader = csv.reader(f)
				data = list(csvreader)[1:]
			ids = sorted([d[0] for d in data])
			train_ids = ids[:round(len(ids)*8/10)]
			val_ids = ids[round(len(ids)*8/10):]
			label_lookup = {k:np.array(v.split(' ')) for k,v in data}

		if self.split == "train":
			self.data = [(os.path.join(self.base_path, "train", i+"_green.png"), label_lookup[i]) for i in train_ids]
			labels = [self.to_onehot(np.array(v.split(' '))) for _, v in data]
			self.class_weights = np.sum(labels, axis=0)
			self.class_weights = self.class_weights.max() / self.class_weights
			self.class_weights = self.class_weights / self.n_classes
			self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)
		elif self.split == "val":
			self.data = [(os.path.join(self.base_path, "train", i+"_green.png"), label_lookup[i]) for i in val_ids]
		elif self.split == "test":
			filelist = glob.glob(os.path.join(self.base_path, "test", "*.png"))
			test_ids = list(set([fn.split('/')[-1].split('_')[0] for fn in filelist]))
			self.data = [(os.path.join(self.base_path, "test", i+"_green.png"), i) for i in test_ids]
		else:
			raise Exception("Invalid dataset split.")

		if debug and self.split != "test":
			self.data = self.data[:100]

	def __getitem__(self, index):

		if self.split in ["train", "val"]:
			fn, label = self.data[index]
			img = Image.open(fn).convert("RGB")
			img = self.transforms(img)
			label = self.to_onehot(label)
			return img, label

		else:
			fn, frame_id = self.data[index]
			img = Image.open(fn).convert("RGB")
			img = self.transforms(img)
			return img, frame_id

	def __len__(self):
		return len(self.data)

	def to_onehot(self, lbl):
		out = np.zeros(self.n_classes)
		for i in lbl:
			out[int(i)] = 1
		return out

	def from_onehot(self, lbl):
		return np.where(lbl.flatten() == 1)[0].tolist()

