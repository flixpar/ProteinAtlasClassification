import os
import sys
import tqdm
import glob
import numpy as np
import importlib.util
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import albumentations as tfms

from loaders.loader import ProteinImageDataset
from models.resnet import Resnet
from models.pretrained import Pretrained
from models.postprocess import postprocess
from models.postprocess import optimize_uniform_threshold, optimize_perclass_threshold

from util.logger import Logger
from util.misc import get_model

def main():

	if not len(sys.argv) == 4:
		raise ValueError("Not enough arguments")

	folder_name = sys.argv[1]
	folder_path = os.path.join("./saves", folder_name)
	if not os.path.exists(folder_path):
		raise ValueError("No matching save folder: {}".format(folder_path))

	save_id = sys.argv[2]
	if os.path.exists(os.path.join(folder_path, "save_{}.pth".format(save_id))):
		save_path = os.path.join(folder_path, "save_{}.pth".format(save_id))
	elif os.path.exists(os.path.join(folder_path, "save_{:03d}.pth".format(int(save_id)))):
		save_path = os.path.join(folder_path, "save_{:03d}.pth".format(int(save_id)))
	else:
		raise Exception("Specified save not found: {}".format(save_id))

	auto_submit = sys.argv[3]
	auto_submit = (auto_submit == "True")

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()

	test_transforms = None
	test_dataset = ProteinImageDataset(split="test", args=args,
		transforms=test_transforms, channels=args.img_channels, debug=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
		num_workers=args.workers, pin_memory=True)

	model = get_model(args)
	state_dict = torch.load(save_path)
	if "module." in list(state_dict.keys())[0]:
		temp_state = OrderedDict()
		for k, v in state_dict.items():
			temp_state[k.split("module.")[-1]] = v
		state_dict = temp_state
	model.load_state_dict(state_dict)
	model.cuda()

	logger = Logger(path=folder_path)

	print("Find Threshold")
	thresh = find_threshold(args, model)

	print("Test")
	test_results = test(args, model, test_loader, thresh)
	logger.write_test_results(test_results, test_dataset.test_ids)

	if auto_submit:
		print("Submitting")
		logger.submit_kaggle()

def test(args, model, test_loader, thresh):
	model.eval()

	outputs = []
	with torch.no_grad():
		for i, (image, frame_id) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):

			image = image.to(dtype=torch.float32).cuda(non_blocking=True)
			output = model(image)
			output = torch.sigmoid(output)
			output = output.cpu().numpy()

			frame_id = frame_id[0]
			outputs.append((frame_id, output))

	preds = [p[1] for p in outputs]
	preds = postprocess(args, preds=preds, threshold=thresh)
	preds = [test_loader.dataset.decode_label(p) for p in preds]

	preds = list(zip([p[0] for p in outputs], preds))
	return preds

def find_threshold(args, model):
	if not ("uniform_thresh" in args.postprocessing or "perclass_thresh" in args.postprocessing):
		return None
	model.eval()
	preds, targets = [], []
	dataset  = ProteinImageDataset(split="val", args=args, channels=args.img_channels)
	loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1, 
		num_workers=args.workers, pin_memory=True)
	with torch.no_grad():
		for (images, labels) in tqdm.tqdm(loader, total=len(loader)):
			images = images.to(dtype=torch.float32).cuda(non_blocking=True)
			pred = torch.sigmoid(model(images)).cpu().numpy()
			labels = labels.cpu().numpy().astype(np.int)
			preds.append(pred)
			targets.append(labels)
	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()
	if "uniform_thresh" in args.postprocessing:
		return optimize_uniform_threshold(preds, targets)
	elif "perclass_thresh" in args.postprocessing:
		return optimize_perclass_threshold(preds, targets)
	else:
		return None

if __name__ == "__main__":
	main()