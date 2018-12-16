import torch
from torch import nn

import numpy as np
from sklearn import metrics

import pystruct
from pystruct.learners import NSlackSSVM
from pystruct.models import MultiLabelClf

def postprocess(args, preds, targets=None, threshold=None):

	if targets is not None:
		if "uniform_thresh" in args.postprocessing:
			threshold = optimize_uniform_threshold(preds, targets)
			threshold = np.full((1, preds.shape[1]), threshold)
		elif "perclass_thresh" in args.postprocessing:
			threshold = optimize_perclass_threshold(preds, targets)
			threshold = threshold.reshape(1, -1)
		else:
			threshold = np.full((1, preds.shape[1]), 0.5)
	else:
		if isinstance(threshold, float):
			threshold = np.full((1, preds.shape[1]), threshold)
		elif isinstance(threshold, np.ndarray):
			threshold = threshold.reshape(1, -1)
		elif isinstance(threshold, list):
			threshold = np.asarray(threshold).reshape(1, -1)

	if "max3" in args.postprocessing:
		num_true = (preds > threshold).astype(np.int).sum()
		if num_true > 3:
			mask = np.argsort(preds, axis=1)[:-3]
			preds[mask] = 0
	elif "max4" in args.postprocessing:
		num_true = (preds > threshold).astype(np.int).sum()
		if num_true > 4:
			mask = np.argsort(preds, axis=1)[:-4]
			preds[mask] = 0

	if "9+10" in args.postprocessing:
		mask_9_10 = np.mean(preds[:, 9:11], axis=1)
		mask_9_10 = mask_9_10 > np.mean(threshold[:, 9:11], axis=1)
		preds[mask_9_10, 9:11]  = 1

	if "min1" in args.postprocessing:
		if np.any(np.all(preds <= threshold, axis=1)):
			mask = np.all(preds <= threshold, axis=1)
			tops = np.argmax(preds, axis=1)
			preds[tops][mask] = 1

	preds = (preds > threshold).astype(np.int)
	return preds

def optimize_uniform_threshold(p_pred, y_true):
	searchspace = np.linspace(0, 1, num=100)
	scores = np.asarray([metrics.f1_score(y_true, (p_pred>t), average="macro") for t in searchspace])
	best = searchspace[scores.argmax()]
	return best

def optimize_perclass_threshold(p_pred, y_true):
	searchspace = np.linspace(0, 1, num=100)
	scores = np.asarray([metrics.f1_score(y_true, (p_pred>t), average=None) for t in searchspace])
	best = searchspace[scores.argmax(axis=0)]
	return best

def crf_postprocess(X_train, y_train, X_test, train_examples=2000):
	clf = NSlackSSVM(MultiLabelClf(), verbose=1, n_jobs=-1, show_loss_every=1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	pred = np.array(pred)
	return pred
