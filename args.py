import torch
import albumentations as tfms

class Args:

	##############################
	###### Hyperparameters #######
	##############################

	epochs = 30                     # DEFAULT 30 (int 1-99)
	initial_lr = 2e-5               # DEFAULT 2e-5 (float)
	batch_size = 16                 # DEFAULT 16 (int)

	trainval_ratio = 0.8            # DEFAULT 0.8
	img_size = None                 # DEFAULT None (None | int 224-1024)
	img_channels = "g"              # DEFAULT g (str {r, g, b, y})

	arch = "resnet152"              # DEFAULT resnet152 (resnet152 | senet154 | inceptionv4)
	weight_mode = "inverse"         # DEFAULT inverse (inverse + sqrt | None)
	loss = "softmargin"             # DEFAULT softmargin (softmargin | focal)

	device_ids = [0,1]              # DEFAULT [0,] (list 0-8)
	workers = 8                     # DEFAULT 8 (int 0-16)

	log_freq = 5                    # DEFAULT 10 (int)
	n_val_samples = 1024            # DEFAULT 1024 (int)

	##############################
	###### Image Transforms ######
	##############################

	train_transforms = tfms.Compose([
		tfms.HorizontalFlip(p=0.5),
		tfms.VerticalFlip(p=0.5),
		tfms.RandomBrightness(),
		tfms.RandomContrast(),
		tfms.Normalize(mean=[0.054, 0.054, 0.054], std=[0.089, 0.089, 0.089])
	])
	test_transforms = tfms.Compose([
		tfms.Normalize(mean=[0.054, 0.054, 0.054], std=[0.089, 0.089, 0.089])
	])

	##############################
	########### Test #############
	##############################

	test_augmentation = None        # DEFAULT None (None)
	
	##############################
	########## Paths #############
	##############################

	datapath = "/home/felix/projects/class/deeplearning/final/data/"
