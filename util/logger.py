import os
import datetime
import pickle
import json
import csv
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


class Logger:

	def __init__(self):
		dt = datetime.datetime.now().strftime("%m%d_%H%M")
		self.path = "./saves/{}".format(dt)
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.losses = []
		self.scores = []
		self.eval_metrics = set()
		self.main_log_fn = os.path.join(self.path, "log.txt")
		shutil.copy2("args.py", self.path)

	def write_test_results(self, results, test_ids):
		out_fn = os.path.join(self.path, "test_results.csv")
		results_lookup = {r[0]:r[1] for r in results}
		with open(out_fn, "w") as f:
			csvwriter = csv.writer(f)
			csvwriter.writerow(("Id", "Predicted"))
			for frame_id in test_ids:
				pred = results_lookup[frame_id] if frame_id in results_lookup else []
				pred = [str(i) for i in pred]
				csvwriter.writerow((frame_id, " ".join(pred)))
		self.print("Results written to: {}\n".format(out_fn))

	def save_model(self, model, epoch):
		if isinstance(epoch, int):
			fn = os.path.join(self.path, "save_{:03d}.pth".format(epoch))
		else:
			fn = os.path.join(self.path, "save_{}.pth".format(epoch))
		torch.save(model.state_dict(), fn)
		self.print("Saved model to: {}\n".format(fn))

	def print(self, *x):
		print(*x)
		self.log(*x)

	def log(self, *x):
		with open(self.main_log_fn, "a") as f:
			print(*x, file=f, flush=True)

	def log_loss(self, l):
		self.losses.append(l)

	def log_eval(self, data):
		self.eval_metrics = set.union(self.eval_metrics, set(data.keys()))
		for k in self.eval_metrics:
			if not k in data:
				data[k] = ''
		self.scores.append(data)

	def save(self):

		with open(os.path.join(self.path, "loss.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it", "loss"])
			csvwriter.writeheader()
			for it, loss in enumerate(self.losses):
				row = {"it": it, "loss": loss}
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "eval.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it"] + sorted(list(self.eval_metrics)))
			csvwriter.writeheader()
			for it, score in enumerate(self.scores):
				score["it"] = it
				csvwriter.writerow(score)

		loss_data = pd.read_csv(os.path.join(self.path, "loss.csv"))
		lossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = loss_data
		)
		lossplot.set_title("Train loss")
		lossplot.figure.savefig(os.path.join(self.path, "train_loss.png"))

		plt.clf()

		eval_data = pd.read_csv(os.path.join(self.path, "eval.csv"))

		if not set(["f1", "loss", "acc"]) <= set(eval_data.columns.values):
			return

		evalplot = eval_data.plot(x="it", y="loss", legend=False, color="b")
		small_ax = evalplot.twinx()
		evalplot = eval_data.plot(x="it", y="f1", legend=False, color="r", ax=small_ax)
		evalplot = eval_data.plot(x="it", y="acc", legend=False, color="g", ax=small_ax)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation Metrics")
		evalplot.figure.savefig(os.path.join(self.path, "eval.png"))

		plt.clf()

		f1plot = sns.lineplot(
			x = "it",
			y = "f1",
			data = eval_data
		)
		f1plot.set_title("Eval Macro F1 Score")
		f1plot.figure.savefig(os.path.join(self.path, "eval_f1.png"))

		plt.clf()

		accplot = sns.lineplot(
			x = "it",
			y = "acc",
			data = eval_data
		)
		accplot.set_title("Eval Accuracy")
		accplot.figure.savefig(os.path.join(self.path, "eval_acc.png"))

		plt.clf()

		evallossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = eval_data
		)
		evallossplot.set_title("Eval Loss")
		evallossplot.figure.savefig(os.path.join(self.path, "eval_loss.png"))

