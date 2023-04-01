import torch
import torchvision
import torch.optim as optim
from dataloader import load_data
from torch.utils import data as data_utils
from tqdm import tqdm
from torchsummary import summary
import argparse
from models.AdaUnet import AdaUNet
from models.AdaNet import AdaNet_v1

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Code to train model')

parser.add_argument("--cams", help="fold index [1-5]", required=True, type=str)

parser.add_argument("--cam_data_path", help="fold index [1-5]", default="./config/cam/", type=str)

parser.add_argument("--batch_size", help="batch size", default=16, type=int)

parser.add_argument('--root_data', help='data folder path', default="../training/training/training/", type=str)

parser.add_argument("--weight_root", help="weight folder", default="/content//gdrive/MyDrive/G/save/", type=str)

parser.add_argument("--model_name", help="name of the weight file", required=True, type=str)

parser.add_argument("--pretrain_weight", help="name of the weight file", required=True, type=str)

args = parser.parse_args()

def train(device, model, train_data_loader, val_data_loader, optimizer, nepochs, WEIGTH_PATH):
	train_losses = []        
	val_losses = []
	min_val_loss = 1000
	early_stop_count = 0
	print("")
	for epoch in range(nepochs):  # loop over the dataset multiple times
		running_loss = 0.0
		print("Epoch {} training".format(epoch))
		prog_bar = tqdm(enumerate(train_data_loader))
		for step, data in prog_bar:
			model.train()
			# get the inputs; data is a list of [inputs, labels]
			inputs, gt = data[0], data[1]
			inputs = inputs.to(device)
			gt = gt.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			#outputs = model(inputs)
			loss = model.dice_loss(X=inputs, Y=gt)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss = ((running_loss * step) + loss.item())/(step+1)
			prog_bar.set_description('Dice loss: {:.4f}'.format(running_loss))
		
		train_losses.append(running_loss)
		val_loss = validate(val_data_loader, epoch, device, model)
		val_losses.append(val_loss)
		
		if val_loss < min_val_loss:
			early_stop_count = 0
			torch.save(model.state_dict(), WEIGTH_PATH)
			print("Saving weights: val loss improved from %.4f to %.4f" % (min_val_loss, val_loss))
			min_val_loss = val_loss
		else:
			early_stop_count = early_stop_count + 1

		if early_stop_count >= 5:
			print(' Training complete due to early stopping')
			break
		
		print("")


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
		running_loss = ((running_loss * step) + val_loss.item())/(step+1)
		
		prog_bar.set_description('Val Dice loss: {:.4f}'.format(running_loss))
		
		loss_list.append(val_loss.item())
	return running_loss


if __name__ == "__main__":
	model_name = args.model_name
	batch_size = args.batch_size
	TRAINING_PATH = args.root_data
	ROOT_WEIGHTPATH = args.weight_root
	cam_data_path = args.cam_data_path
	cams = args.cams
	
	WEIGTH_PATH = ROOT_WEIGHTPATH + model_name + ".pth"

	# Dataset and Dataloader setup
	train_dataset = load_data(cams=cams, cam_file_path=cam_data_path, TRAINING_PATH=TRAINING_PATH,  mode=0)
	val_dataset = load_data(cams=cams, cam_file_path=cam_data_path, TRAINING_PATH=TRAINING_PATH, mode=1)

	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True)

	val_data_loader = data_utils.DataLoader(
		val_dataset, batch_size=batch_size)

	device = torch.device("cuda" if use_cuda else "cpu")

	# Model
	Ada_model = AdaNet_v1()
	model = AdaUNet(Ada_model).to(device)
	msg = model.load_state_dict(torch.load(ROOT_WEIGHTPATH+args.pretrain_weight+".pth"),  strict=False)
	print(f'Loading messages: \n {msg}')
	for p in model.parameters():
		print(p.name)

	#summary(model, (1, 224, 224))
	#print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

	optimizer = torch.optim.AdamW(model.parameters())

	# Train!
	train(device, model, train_data_loader, val_data_loader, optimizer, nepochs=100, WEIGTH_PATH=WEIGTH_PATH)