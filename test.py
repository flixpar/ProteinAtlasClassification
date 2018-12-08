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
from torchvision import transforms as tfms

from loaders.loader import ProteinImageDataset
from models.resnet import Resnet
from models.pretrained import Pretrained

from util.logger import Logger
from util.misc import get_model

def main():

	if not len(sys.argv) == 3:
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

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()
	
	test_transforms = tfms.Compose([
		tfms.ToTensor(),
		tfms.Normalize(mean=[0.054, 0.054, 0.054], std=[0.089, 0.089, 0.089])
	])

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

	logger = Logger()

	print("Test")
	test_results = test(model, test_loader)
	logger.write_test_results(test_results, test_dataset.test_ids)

def test(model, test_loader):
	model.eval()

	preds = []
	with torch.no_grad():
		for i, (image, frame_id) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):

			image = image.to(dtype=torch.float32).cuda(non_blocking=True)
			output = model(image)
			output = torch.sigmoid(output)
			output = output.cpu().numpy()

			pred = (output > 0.5).astype(np.int)
			if not np.any(pred):
				top = np.argmax(pred, axis=1)
				pred = np.zeros(pred.shape)
				pred[:, top] = 1
			pred = test_loader.dataset.from_onehot(pred)

			frame_id = frame_id[0]
			preds.append((frame_id, pred))

	return preds

if __name__ == "__main__":
	main()