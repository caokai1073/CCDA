import torch
from torchvision import datasets, transforms
import os

def get_svhn(dataset_root, batch_size, train):

	#image pre-processing
	pre_process = transforms.Compose([transforms.Resize(32),
									#(H,W,C)->(C,H,W), (0,1)
									transforms.ToTensor(),
									#(-1,1)
									transforms.Normalize(
										mean=(0.5,0.5,0.5),
										std=(0.5,0.5,0.5)
									)])

	if train:
		svhn_dataset = datasets.SVHN(root=os.path.join(dataset_root,'svhn'),
									split='train',
									transform=pre_process,
									download=True)
	else:
		svhn_dataset = datasets.SVHN(root=os.path.join(dataset_root,'svhn'),
									split='test',
									transform=pre_process,
									download=True)

	svhn_data_loader = torch.utils.data.DataLoader(
		dataset=svhn_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True)

	return svhn_data_loader