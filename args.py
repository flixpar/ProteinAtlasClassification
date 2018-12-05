class Args:

	##############################
	###### Hyperparameters #######
	##############################

	epochs = 30                     # DEFAULT 30 (int 1-99)
	initial_lr = 2e-5               # DEFAULT 2e-5 (float)
	batch_size = 16                 # DEFAULT 16 (int)

	trainval_ratio = 0.8            # DEFAULT 0.8
	img_size = 512                  # DEFAULT 512 (int 256-1024)
	img_channels = "g"              # DEFAULT g (str {r, g, b, y})

	arch = "resnet152"              # DEFAULT resnet152 (resnet152 | senet154 | inceptionv4)
	weight_mode = "inverse"         # DEFAULT inverse (inverse + sqrt | None)
	loss = "softmargin"             # DEFAULT softmargin (softmargin | focal)

	device_ids = [0,1]              # DEFAULT [0,] (list 0-8)
	workers = 6                     # DEFAULT 6 (int 0-12)

	log_freq = 5                    # DEFAULT 10 (int)

	##############################
	########### Test #############
	##############################

	test_augmentation = None        # DEFAULT None (None)
	test_model_path = "latest"      # DEFAULT latest (latest | <path>)

	##############################
	########## Paths #############
	##############################

	datapath = "/home/felix/projects/class/deeplearning/final/data/"
