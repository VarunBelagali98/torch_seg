import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
import math

class load_data(torch.utils.data.Dataset):
	def __init__(self,file_path,TRAINING_PATH, mode=0):
		random.seed(0)
		self.TRAINING_PATH = TRAINING_PATH

		datalist = list(np.genfromtxt(file_path,dtype='str'))


		random.shuffle(datalist)

		if len(datalist) > 6000:
			datalist = random.sample(list(datalist), 6000)


		if mode == 0:
			datalist = datalist[:-500]
		if mode == 1:
			datalist = datalist[-500:]
		
		#random.shuffle(images_indx)
		self.datalist = datalist

	def __len__(self):
		return int(len(self.datalist))

	def __getitem__(self, idx):
		s = 224
		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '.png'
		img_in = cv2.imread(fname)
		img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
		img_in = cv2.resize(img_in, ( s , s ))

		fname = self.TRAINING_PATH + str(self.datalist[idx]) + '_seg.png'
		img_g = cv2.imread(fname)
		img_g = cv2.resize(img_g, ( s , s )) 
			
		im = img_g[:,:,0]/255
		xaxis = np.sum(im,axis=0)
		yaxis = np.sum(im,axis=1)
		xs = np.nonzero(xaxis)[0]
		ys = np.nonzero(yaxis)[0]

		negative_mask = np.ones((s, s, 1))
		positive_mask = np.zeros((s, s, 1))

		if ((len(xs)<1) or (len(ys)<1)):
			x0,x1,y0,y1 = 0,0,0,0
		else:
			x0 = xs[0]
			x1 = xs[-1]
			y0 = ys[0]
			y1 = ys[-1]
	
			positive_mask[y0:y1+1, x0:x1+1, :] = 1
			negative_mask[y0:y1+1, x0:x1+1, :] = 0
			
		img =img_in[:,:,np.newaxis]/255.0
		#pos = positive_mask[:,:,:]
		#neg = negative_mask[:,:,:]
		img_ann = img_g[:,:,1,np.newaxis]/255.0

		img = np.transpose(img, (2, 0, 1))
		img_ann = np.transpose(img_ann, (2, 0, 1))

		img = torch.FloatTensor(img)
		img_ann = torch.FloatTensor(img_ann)

		return (img, img_ann)

