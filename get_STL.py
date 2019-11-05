import torch
from torchvision import datasets, transforms
import os

def get_STL(dataset_root, batch_size, train):
	pre_proscess = transforms.Compose([transforms.Resize(32),
										transforms.ToTensor(),
										transforms.Normalize(
											mean=(0.5,0.5,0.5),
											std=(0.5,0.5,0.5)
									)])

	STL_dataset = datasets.STL10(root=os.path.join(dataset_root, 'STL'),
										split=train,
										transform=pre_proscess,
										download=True)

	STL_data_loader = torch.utils.data.DataLoader(
		dataset=STL_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=8)

	return STL_data_loader