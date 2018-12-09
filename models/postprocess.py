import torch
from torch import nn
import numpy as np

import pystruct
from pystruct.learners import NSlackSSVM
from pystruct.models import MultiLabelClf

def postprocess(pred):
	threshold = np.full((1, pred.shape[1]), 0.5)

	num_true = (pred > threshold).astype(np.int).sum()
	if num_true > 3:
		mask = np.argsort(pred, axis=1)[:-3]
		pred[mask] = 0

	pred = (pred > threshold).astype(np.int)

	if not np.any(pred):
		top = np.argmax(pred, axis=1)
		pred = np.zeros(pred.shape)
		pred[:, top] = 1

	mask_9_10  = np.logical_or((pred[:,9] == 1), (pred[:,10] == 1))
	pred[mask_9_10, 9]  = 1
	pred[mask_9_10, 10] = 1

	return pred

def crf_postprocess(X_train, y_train, X_test, train_examples=2000):
	clf = NSlackSSVM(MultiLabelClf(), verbose=1, n_jobs=-1, show_loss_every=1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	pred = np.array(pred)
	return pred
