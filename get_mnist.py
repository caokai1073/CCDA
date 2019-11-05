import torch
from torchvision import datasets, transforms
import os

def get_mnist(dataset_root, batch_size, train):
	pre_proscess = transforms.Compose([transforms.Resize(32),
										transforms.ToTensor(),
										transforms.Normalize(
											mean=(0.1307, ),
											std=(0.3081, )
									)])

	mnist_dataset = datasets.MNIST(root=os.path.join(dataset_root, 'mnist'),
									train=train,
									transform=pre_proscess,
									download=True)

	mnist_data_loader = torch.utils.data.DataLoader(
		dataset=mnist_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=8)

	return mnist_data_loader