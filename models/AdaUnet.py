import torch
from torch import nn
from torch.nn import functional as F
import math
from .conv import Conv2dTranspose, Conv2d
import numpy as np

class AdaUNet(nn.Module):
	def __init__(self, AdaNet=None, base_model=None):
		super(AdaUNet, self).__init__()

		self.adapter = AdaNet
		self.base_model = base_model 


	def forward(self, X):
		X =  self.adapter(X)
		X = self.base_model(X)
		return X

	def adapter_out(self, x):
		x = self.adapter(x)
		return x

	def dice_loss(self, X=None, Y=None):
		pred = self.forward(X)
		smooth = 1.
		y_true_f = torch.reshape(Y, (-1,))
		y_pred_f = torch.reshape(pred, (-1,))
		intersection = torch.sum(y_true_f * y_pred_f)
		score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
		return (1-score)

	def dice_score(self, X, Y):
		preds = self.forward(X)
		preds = preds.cpu().detach().numpy()
		Y = Y.cpu().detach().numpy()
		batch_size = preds.shape[0]
		dices = []

		preds = preds > 0.5
		preds = preds.astype(int)

		for i in range(0, batch_size):
			pred = preds[i, :, :, :]
			gt = Y[i, :, :, :]
			if (np.sum(gt) == 0 and np.sum(pred) == 0):
				dice_val = 1
				iou=1
			else:
				dice_val = 2.0 * np.sum(np.multiply(pred,gt))/(np.sum(pred)+np.sum(gt)+0.000000000000001)
				iou = np.sum(np.multiply(pred,gt))/(np.sum(pred)+np.sum(gt)- np.sum(np.multiply(pred,gt)) +0.000000000000001)
			dices.append(dice_val)
		return dices
