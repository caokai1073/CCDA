import torch
from torchvision import datasets, transforms
import os

def get_cifar(dataset_root, batch_size, train):
	pre_proscess = transforms.Compose([transforms.Resize(32),
										transforms.ToTensor(),
										transforms.Normalize(
											mean=(0.5,0.5,0.5),
											std=(0.5,0.5,0.5)
									)])

	cifar_dataset = datasets.CIFAR10(root=os.path.join(dataset_root, 'cifar'),
										train=train,
										transform=pre_proscess,
										download=True)

	cifar_data_loader = torch.utils.data.DataLoader(
		dataset=cifar_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=8)

	return cifar_data_loader