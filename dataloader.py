import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import random
import math
from config.norm import norm_dict

class load_data(torch.utils.data.Dataset):
	def __init__(self,cams, cam_file_path,TRAINING_PATH, mode=0, train_cams=None):
		random.seed(0)
		self.TRAINING_PATH = TRAINING_PATH

		datalist = self.get_datalist(cams, cam_file_path, mode)
		print(len(datalist))
		self.datalist = datalist
		print(mode, self.datalist)
		self.mode = mode
		if train_cams == None:
			train_cams =cams
		self.mean = norm_dict[train_cams]["mean"]
		self.std = norm_dict[train_cams]["std"]

		self.intensity_scales = np.array([0.5, 0.8, 1, 1.2, 1.5])

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

		if self.mode == 0 or self.mode == 1:
			scale = np.random.choice(a=self.intensity_scales, size=1, p=[0.1, 0.1, 0.6, 0.1, 0.1])[0]
			img_in = img_in * scale
			img_in = np.clip(img_in, a_min=0, a_max=255) 
			
		img = (img_in[:,:,np.newaxis] - self.mean) / self.std
		#pos = positive_mask[:,:,:]
		#neg = negative_mask[:,:,:]
		img_ann = img_g[:,:,1,np.newaxis]/255.0

		img = np.transpose(img, (2, 0, 1))
		img_ann = np.transpose(img_ann, (2, 0, 1))

		img = torch.FloatTensor(img)
		img_ann = torch.FloatTensor(img_ann)

		return (img, img_ann)

	def get_datalist(self, cams, cam_file_path, mode=0):

		camera_map = {
			'1': "KayPentaxHSV9710(Photron)",
			'2': "Phantomv210",
			'3': "HERS5562EndocamWolf",
			'4': "KayPentaxHSV9700(Photron)",
			'5': "FASTCAMMiniAX100type540K-C-16GB"
		}

		cams = cams.split(",")

		train_datalist = []
		val_datalist = []
		test_datalist = []

		for cam in cams:
			random.seed(0)
			path = cam_file_path + camera_map[cam] + "_train" + ".txt"
			cam_datalist = list(np.genfromtxt(path,dtype='str'))
			random.shuffle(cam_datalist)
			if len(cam_datalist) >= 1000:
				train_datalist.extend(cam_datalist[:len(cam_datalist)-500])
				val_datalist.extend(cam_datalist[-500:])
			else:
				val_len = int(0.2 * len(cam_datalist))
				train_datalist.extend(cam_datalist[:len(cam_datalist)-val_len])
				val_datalist.extend(cam_datalist[-val_len:])


			path = cam_file_path + camera_map[cam] + "_test" + ".txt"
			cam_datalist = list(np.genfromtxt(path,dtype='str'))
			test_datalist.extend(cam_datalist)

		if mode == 0:
			return train_datalist
		elif mode == 1:
			return val_datalist
		else:
			return test_datalist


