import os
import datetime
import pickle
import json
import csv
import torch

class Logger:

	def __init__(self):
		dt = datetime.datetime.now().strftime("%m%d_%H%M")
		self.path = "./saves/{}".format(dt)
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.losses = []
		self.scores = []

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
		print("Results written to:", out_fn)

	def save_model(self, model, epoch):
		fn = os.path.join(self.path, "save_{:02d}.pth".format(epoch))
		torch.save(model.state_dict(), fn)