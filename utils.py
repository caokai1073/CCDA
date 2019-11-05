import os
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from get_svhn import get_svhn
from get_mnist import get_mnist
from mnistm import get_mnistm
from get_cifar import get_cifar
from get_STL import get_STL


def init_random_seed(manual_seed):
	seed = None
	if manual_seed is None:
		seed = random.randint(1,10000)
	else:
		seed = manual_seed
	print("use random seed: {}".format(seed))
	random.seed(seed)
	torch.manual_seed(seed) #cpu seed
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)  #gpu seed


def get_data_loader(name, dataset_root, batch_size, train=True):
	if name == "svhn":
		return get_svhn(dataset_root, batch_size, train)
	if name == "mnist":
		return get_mnist(dataset_root, batch_size, train)
	if name == "mnistm":
		return get_mnistm(dataset_root, batch_size, train)	
	if name == "cifar":
		return get_cifar(dataset_root, batch_size, train)
	if name == "STL" and train:
		return get_STL(dataset_root, batch_size, 'train')
	if name == "STL" and train==False:
		return get_STL(dataset_root, batch_size, 'test')


def init_model(net, restore):
	if restore is not None and os.path.exits(restore):
		net.load_state_dict(torch.load(restore))
		net.restored = True
		print("Restore model from: {}".format(os.path.abspath(restore)))
	else:
		print("No trained model, train from scratch.")

	if torch.cuda.is_available():
		cudnn.benchmark =True
		net.cuda()

	return net


def save_model(net, model_root, filename):

	if not os.path.exists(model_root):
		os.makedirs(model_root)
	torch.save(net.state_dict(), os.path.join(model_root, filename))
	print("save pretrained model to: {}".format(os.path.join(model_root, filename)))
