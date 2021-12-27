import torch
import torchvision
from models.unet import UNet
import torch.optim as optim
from dataloader import load_data
from torch.utils import data as data_utils
from tqdm import tqdm
from torchsummary import summary
import argparse

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Code to train model')

parser.add_argument("--fold", help="fold index [1-5]", required=True, type=int)

parser.add_argument("--batch_size", help="batch size", default=32, type=int)

parser.add_argument('--root_data', help='data folder path', default="../training/training/training/", type=str)

parser.add_argument('--fold_files', help='data folder path', default="../fold_files/annfiles_fold", type=str)

parser.add_argument("--per", help="Percentage of data to be used", default=None, type=float)

parser.add_argument("--weight_root", help="weight folder", default="/content//gdrive/MyDrive/colab-data/weights/", type=str)

parser.add_argument("--model_name", help="name of the weight file", required=True, type=str)

args = parser.parse_args()

def train(device, model, trainloader, valloader, optimizer, nepochs, WEIGTH_PATH):
	train_losses = []        
	val_losses = []
	print("")
	for epoch in range(nepochs):  # loop over the dataset multiple times
		running_loss = 0.0
		print("Epoch {} training".format(epoch))
		prog_bar = tqdm(enumerate(train_data_loader))
		for step, data in prog_bar:
			# get the inputs; data is a list of [inputs, labels]
			inputs, gt = data[0], data[1]
			inputs = inputs.to(device)
			gt = gt.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			#outputs = model(inputs)
			loss = model.dice_loss(inputs, gt)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss = running_loss + loss.item()
			prog_bar.set_description('Dice loss: {}'.format(running_loss / (step + 1)))
		
		train_losses.append(running_loss)
		val_loss = validate(val_data_loader, epoch, device, model)
		val_losses.append(val_loss)
		save_check = True

		if save_check == True:
			torch.save(model.state_dict(), WEIGTH_PATH)
		print("")


	print(' Training complete')

def validate(val_data_loader, epoch, device, model):
	running_loss = 0
	step = 0
	print("Epoch {} validation".format(epoch))
	prog_bar = tqdm(enumerate(val_data_loader))
	loss_list = []
	for step, data in prog_bar:
		# Move data to CUDA device
		inputs, gt = data[0], data[1]
		inputs = inputs.to(device)
		gt = gt.to(device)
		
		model.eval()
		val_loss = model.dice_loss(inputs, gt)
		running_loss = running_loss + val_loss.item()
		
		prog_bar.set_description('Val Dice loss: {}'.format(running_loss / (step + 1)))
		
		loss_list.append(val_loss.item())
	return sum(loss_list)/len(loss_list)


if __name__ == "__main__":
	fold = args.fold
	per = args.per
	seed_select = None
	model_name = args.model_name
	batch_size = args.batch_size
	TRAINING_PATH = args.root_data
	FOLD_PATH = args.fold_files
	ROOT_WEIGHTPATH = args.weight_root
	
	WEIGTH_PATH = ROOT_WEIGHTPATH + model_name + ".pth"

	# Dataset and Dataloader setup
	train_dataset = load_data(fold, 0, per, seed_select=seed_select, TRAINING_PATH=TRAINING_PATH, FOLD_PATH=FOLD_PATH)
	val_dataset = load_data(fold, 1, per, seed_select=seed_select, TRAINING_PATH=TRAINING_PATH, FOLD_PATH=FOLD_PATH)

	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=False)

	val_data_loader = data_utils.DataLoader(
		val_dataset, batch_size=batch_size)

	device = torch.device("cuda" if use_cuda else "cpu")

	# Model
	model = UNet().to(device)
	summary(model, (1, 224, 224))
	print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

	# Train!
	train(device, model, train_data_loader, val_data_loader, optimizer, nepochs=10, WEIGTH_PATH=WEIGTH_PATH)