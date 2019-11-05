import os
import sys

# sys.path.append('../')
import torch
import torch.nn as nn
from model import CorA_model, Discriminator
from train import train_model
from utils import get_data_loader, init_model, init_random_seed
from test import eval

class Config(object):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	dataset_root = os.path.expanduser(os.path.join('~', 'Datasets'))
	model_root = os.path.expanduser(os.path.join('~','Models', 'pytorch-svhn'))

	batch_size = 128

	src_dataset = "mnist"
	#model_trained = True
	#classifier_restore = os.path.join(model_root, dataset + '-classifier.pt')

	tgt_dataset = "svhn"

	num_epochs = 300
	log_step = 20
	save_step = 10
	eval_step = 2

	manual_seed = 8888
	alpha = 0

	lr = 2e-4

params = Config()

init_random_seed(params.manual_seed)

src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=True)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=False)

net = CorA_model()
net =  nn.DataParallel(net)
CorA = init_model(net, restore = None)
net = Discriminator()
net =  nn.DataParallel(net)
Discriminator = init_model(net, restore = None)
# net = tgt_net()
# net =  nn.DataParallel(net)
# tgt_net = init_model(net, restore = None)

print("Training CorA and Discriminator model")

# CorA.load_state_dict(torch.load("CorA-190.pt"))
# Discriminator.load_state_dict(torch.load("Discriminator-190.pt"))
if True:
	CorA, Discriminator = train_model(CorA, Discriminator, params, src_data_loader, \
		src_data_loader_eval, tgt_data_loader, tgt_data_loader_eval)
else:	
	print("Evaluating CorA for source domain {}".format(params.src_dataset))
	eval(CorA, Discriminator, src_data_loader_eval, params.device, "src")
	print("Evaluating CorA for target domain {}".format(params.tgt_dataset))
	eval(CorA, Discriminator, tgt_data_loader_eval, params.device, "tgt")

	# train_tgt(CorA, tgt_net,tgt_data_loader, tgt_data_loader_eval, params.device)
