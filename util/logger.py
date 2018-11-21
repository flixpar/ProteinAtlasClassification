import os
import datetime
import pickle
import json
import csv
import torch
import numpy as np
import pandas as pd
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

	def write_test_results(self, results, test_ids):
		out_fn = os.path.join(self.path, "test_results.csv")
		results_lookup = {r[0]:r[1] for r in results}
		with open(out_fn, "w") as f:
			csvwriter = csv.writer(f)
			csvwriter.writerow(("Id", "Predicted"))
			for frame_id in test_ids:
				pred = results_lookup[frame_id]
				pred = [str(i) for i in pred]
				csvwriter.writerow((frame_id, " ".join(pred)))
		self.print("Results written to:", out_fn)

	def save_model(self, model, epoch):
		if isinstance(epoch, int):
			fn = os.path.join(self.path, "save_{:03d}.pth".format(epoch))
		else:
			fn = os.path.join(self.path, "save_{}.pth".format(epoch))
		torch.save(model.state_dict(), fn)
		self.print("Saved model to: {}".format(fn))

	def print(self, *x):
		print(*x)
		with open(self.main_log_fn, "a") as f:
			for y in x:
				f.write(y)
			f.write("\n")

	def log(self, *x):
		with open(self.main_log_fn, "a") as f:
			for y in x:
				f.write(y)
			f.write("\n")

	def log_loss(self, l):
		self.losses.append(l)

	def log_eval(self, data):
		self.eval_metrics += set(data.keys())
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
		lossplot.savefig("train_loss.png")

		eval_data = pd.read_csv(os.path.join(self.path, "eval.csv"))
		evalplot = sns.lineplot(
			x = "it",
			y = list(self.eval_metrics),
			data = eval_data
		)
		evalplot.savefig("eval.png")
